use crate::shader::*;
use alloc::borrow::Cow;
use alloc::sync::Arc;
use bevy_asset::AssetId;
use bevy_platform::collections::{hash_map::EntryRef, HashMap, HashSet};
use core::hash::Hash;
use thiserror::Error;
use tracing::debug;
use tracing::warn;

pub(crate) fn wesl_module_path(import_path: &ShaderImport) -> Option<wesl::syntax::ModulePath> {
    match import_path {
        ShaderImport::Custom(name) => {
            let mut segments = name.split("::");
            let package = segments.next().filter(|s| !s.is_empty())?;
            let components = segments
                .map(|s| (!s.is_empty()).then(|| s.to_string()))
                .collect::<Option<Vec<_>>>()?;
            Some(wesl::syntax::ModulePath {
                origin: wesl::syntax::PathOrigin::Package(package.to_string()),
                components,
            })
        }
        // `ModulePath::from_path` would strip anything after a `.` as an extension.
        ShaderImport::AssetPath(path) => {
            let components: Vec<String> = path
                .split('/')
                .filter(|component| !component.is_empty())
                .map(str::to_string)
                .collect();
            (!components.is_empty()).then_some(wesl::syntax::ModulePath {
                origin: wesl::syntax::PathOrigin::Absolute,
                components,
            })
        }
    }
}

fn is_module_not_found(error: &wesl::Error) -> bool {
    match error {
        wesl::Error::ResolveError(wesl::error::ResolveError::ModuleNotFound(..))
        | wesl::Error::ImportError(wesl::error::ImportError::ResolveError(
            wesl::error::ResolveError::ModuleNotFound(..),
        )) => true,
        wesl::Error::Error(diagnostic) => is_module_not_found(&diagnostic.error),
        _ => false,
    }
}

/// Fully composed source code of a shader module, with all shader defs applied.
///
/// This is roughly equivalent to [`wgpu::ShaderSource`](https://docs.rs/wgpu/latest/wgpu/enum.ShaderSource.html),
/// but with less variants and more concrete types instead of [`Cow`].
///
/// This source will be parsed and validated by the renderer.
///
/// Any necessary shader translation (e.g. from WGSL to SPIR-V or vice versa)
/// must be done internally by the renderer.
#[derive(Clone, Debug)]
pub enum ShaderCacheSource<'a> {
    /// SPIR-V module represented as a slice of words.
    SpirV(&'a [u8]),
    /// WGSL module as a string slice.
    Wgsl(String),
}

/// An id of a pipeline, typically in the [`PipelineCache`](https://docs.rs/bevy/latest/bevy/render/render_resource/struct.PipelineCache.html)
/// Typically corresponds to a unique combination of [`Shader`] and [`ShaderDefVal`]s.
pub type CachedPipelineId = usize;

struct ShaderData<ShaderModule> {
    pipelines: HashSet<CachedPipelineId>,
    processed_shaders: HashMap<Box<[ShaderDefVal]>, Arc<ShaderModule>>,
    resolved_imports: HashMap<ShaderImport, AssetId<Shader>>,
    dependents: HashSet<AssetId<Shader>>,
}

impl<T> Default for ShaderData<T> {
    fn default() -> Self {
        Self {
            pipelines: Default::default(),
            processed_shaders: Default::default(),
            resolved_imports: Default::default(),
            dependents: Default::default(),
        }
    }
}

/// A cache for shaders and shader imports, with asset state-tracking for
/// waiting to load shaders until all imports are resolved.
///
/// Note that the `RenderDevice` generic parameter is a means by which
/// to avoid a cyclic dependency with `bevy_render`, while also permitting
/// alternative rendering implementations. The actual processing of the
/// shader source into a usable compiled module is left to the renderer.
pub struct ShaderCache<ShaderModule, RenderDevice> {
    device: RenderDevice,
    data: HashMap<AssetId<Shader>, ShaderData<ShaderModule>>,
    load_module: fn(
        &RenderDevice,
        ShaderCacheSource,
        &ValidateShader,
    ) -> Result<ShaderModule, ShaderCacheError>,
    module_path_to_asset_id: HashMap<wesl::syntax::ModulePath, AssetId<Shader>>,
    shaders: HashMap<AssetId<Shader>, Arc<Shader>>,
    import_path_shaders: HashMap<ShaderImport, AssetId<Shader>>,
    waiting_on_import: HashMap<ShaderImport, Vec<AssetId<Shader>>>,
    missing_import_logged: HashSet<AssetId<Shader>>,
    /// How often [`Self::clear`] reached each shader. [`Self::get_shared`] compares it before and
    /// after composing outside the lock, to tell whether the variant it made is already stale.
    changes: HashMap<AssetId<Shader>, u64>,
}

/// A compile time shader value definition to be inlined into the shader source.
/// Variant tuples contain the name of the definition, and the value.
#[expect(missing_docs, reason = "Enum variants are self-explanatory")]
#[derive(serde::Serialize, serde::Deserialize, Clone, PartialEq, Eq, Debug, Hash)]
pub enum ShaderDefVal {
    Bool(Cow<'static, str>, bool),
    Int(Cow<'static, str>, i32),
    UInt(Cow<'static, str>, u32),
}

impl From<&'static str> for ShaderDefVal {
    fn from(key: &'static str) -> Self {
        ShaderDefVal::Bool(key.into(), true)
    }
}

impl From<String> for ShaderDefVal {
    fn from(key: String) -> Self {
        ShaderDefVal::Bool(key.into(), true)
    }
}

impl ShaderDefVal {
    /// Returns the value of the define as a string.
    pub fn value_as_string(&self) -> String {
        match self {
            ShaderDefVal::Bool(_, def) => def.to_string(),
            ShaderDefVal::Int(_, def) => def.to_string(),
            ShaderDefVal::UInt(_, def) => def.to_string(),
        }
    }
}

impl<ShaderModule, RenderDevice> ShaderCache<ShaderModule, RenderDevice> {
    /// Creates a new [`ShaderCache`] with the given shader module loading
    /// function. `load_module` is responsible for actually compiling shader
    /// source into a module usable by the render device.
    pub fn new(
        device: RenderDevice,
        load_module: fn(
            &RenderDevice,
            ShaderCacheSource,
            &ValidateShader,
        ) -> Result<ShaderModule, ShaderCacheError>,
    ) -> Self {
        Self {
            device,
            load_module,
            data: Default::default(),
            module_path_to_asset_id: Default::default(),
            shaders: Default::default(),
            import_path_shaders: Default::default(),
            waiting_on_import: Default::default(),
            missing_import_logged: Default::default(),
            changes: Default::default(),
        }
    }

    /// Attempts to retrieve or create a compiled shader module for the given
    /// shader id and shader definitions.
    ///
    /// The provided `pipeline` is tracked so it may later be reported "dirty"
    /// when a shader is removed or replaced.
    ///
    /// Note that the cache is keyed by `id` and `shader_defs`, meaning providing
    /// the same `shader_defs` in a different order, or with redundancies, will
    /// not result in cache hits, and thus require re-composing the module and
    /// calling `load_module` again.
    pub fn get(
        &mut self,
        pipeline: CachedPipelineId,
        id: AssetId<Shader>,
        shader_defs: &[ShaderDefVal],
    ) -> Result<Arc<ShaderModule>, ShaderCacheError> {
        let shader = match self.prepare(pipeline, id, shader_defs)? {
            Prepared::Cached(module) => return Ok(module),
            Prepared::Compose(shader) => shader,
        };
        debug!(
            "processing shader {}, with shader defs {:?}",
            id, shader_defs
        );
        let (source, deps) = compose(
            &self.shaders,
            &self.import_path_shaders,
            &self.module_path_to_asset_id,
            id,
            &shader,
            shader_defs,
        )
        .map_err(|e| self.compose_error(id, &shader, e))?;
        let module = (self.load_module)(
            &self.device,
            source.into_source(&shader),
            &shader.validate_shader,
        )?;
        Ok(self.store(id, shader_defs, Arc::new(module), deps))
    }

    /// Like [`Self::get`], but holds the cache's lock only to look up and to store the module.
    ///
    /// The shader is composed outside the lock, so pipeline tasks can compose in parallel. The module
    /// is created and stored under `create`, a lock of its own: creating it pushes and pops a wgpu
    /// validation error scope, which is global to the device unless wgpu's `std` feature is on.
    /// Two tasks may compose the same variant at once; its module is created only once.
    ///
    /// A composition that [`Self::set_shader`] or [`Self::remove`] made stale meanwhile is never
    /// cached: it is handed back, as `get` would have, while the change queues its pipelines again.
    pub fn get_shared(
        cache: &std::sync::Mutex<Self>,
        create: &std::sync::Mutex<()>,
        pipeline: CachedPipelineId,
        id: AssetId<Shader>,
        shader_defs: &[ShaderDefVal],
    ) -> Result<Arc<ShaderModule>, ShaderCacheError>
    where
        RenderDevice: Clone,
    {
        let lock = || {
            cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        };
        loop {
            let (changes, shader, shaders, imports, modules, device, load_module) = {
                let mut c = lock();
                let shader = match c.prepare(pipeline, id, shader_defs)? {
                    Prepared::Cached(module) => return Ok(module),
                    Prepared::Compose(shader) => shader,
                };
                (
                    c.changes.clone(),
                    shader,
                    c.shaders.clone(),
                    c.import_path_shaders.clone(),
                    c.module_path_to_asset_id.clone(),
                    c.device.clone(),
                    c.load_module,
                )
            };
            debug!(
                "processing shader {}, with shader defs {:?}",
                id, shader_defs
            );
            let composed = compose(&shaders, &imports, &modules, id, &shader, shader_defs);
            // Modules are created and stored under `create`, so a variant another task made while
            // this one was composing is found here, and its module is not created twice.
            let _creating = create
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let (source, deps) = {
                let mut c = lock();
                let stale = c.changes.get(&id) != changes.get(&id);
                let (source, deps) = composed.map_err(|e| match e {
                    // Not logged: the shader changed meanwhile, and its new source is logged when
                    // its pipelines, queued again, compose it.
                    ComposeError::ImportNotFound(_) if stale => {
                        ShaderCacheError::ShaderImportNotYetAvailable
                    }
                    e => c.compose_error(id, &shader, e),
                })?;
                if let Some(module) = c
                    .data
                    .get(&id)
                    .and_then(|data| data.processed_shaders.get(shader_defs))
                {
                    return Ok(module.clone());
                }
                (source, deps)
            };
            let module = load_module(
                &device,
                source.into_source(&shader),
                &shader.validate_shader,
            )?;
            let mut c = lock();
            let changed = |c: &Self, id: &AssetId<Shader>| c.changes.get(id) != changes.get(id);
            if changed(&c, &id) {
                // `clear` reached this shader meanwhile and queued its pipelines again. `get` would
                // have stored this variant, then had it cleared: keep its dependents, not the module.
                c.add_dependents(id, &deps);
                return Ok(Arc::new(module));
            }
            if deps.iter().any(|dep| changed(&c, dep)) {
                // A dependency changed before `clear` could reach this shader through it (dependents
                // are recorded when a variant is stored). Record them, so that every further change
                // reaches this shader as above, and compose again from the new sources.
                c.add_dependents(id, &deps);
                continue;
            }
            return Ok(c.store(id, shader_defs, Arc::new(module), deps));
        }
    }

    /// Under the lock: the shader's imports resolved, the pipeline tracked, and the module if it is made.
    fn prepare(
        &mut self,
        pipeline: CachedPipelineId,
        id: AssetId<Shader>,
        shader_defs: &[ShaderDefVal],
    ) -> Result<Prepared<ShaderModule>, ShaderCacheError> {
        let shader = self
            .shaders
            .get(&id)
            .ok_or(ShaderCacheError::ShaderNotLoaded(id))?
            .clone();

        let data = self.data.entry(id).or_default();

        // Wesl imports are scanned as both module and item candidates.
        if !matches!(shader.source, Source::Wesl(_)) {
            let n_asset_imports = shader
                .imports
                .iter()
                .filter(|import| matches!(import, ShaderImport::AssetPath(_)))
                .count();
            let n_resolved_asset_imports = data
                .resolved_imports
                .keys()
                .filter(|import| matches!(import, ShaderImport::AssetPath(_)))
                .count();
            if n_asset_imports != n_resolved_asset_imports {
                return Err(ShaderCacheError::ShaderImportNotYetAvailable);
            }
        }

        data.pipelines.insert(pipeline);
        Ok(match data.processed_shaders.get(shader_defs) {
            Some(module) => Prepared::Cached(module.clone()),
            None => Prepared::Compose(shader),
        })
    }

    /// Under the lock: a composition's error logged (an unresolved import once per shader).
    fn compose_error(
        &mut self,
        id: AssetId<Shader>,
        shader: &Shader,
        error: ComposeError,
    ) -> ShaderCacheError {
        match error {
            ComposeError::ImportNotFound(diagnostic) => {
                if self.missing_import_logged.insert(id) {
                    warn!(
                        "Shader `{}` has an unresolved import:\n{}",
                        shader.path, diagnostic
                    );
                }
                ShaderCacheError::ShaderImportNotYetAvailable
            }
            ComposeError::Other(e) => e,
        }
    }

    /// Under the lock: the module kept for its variant (unless another task stored it first) and the
    /// shader's WESL dependencies recorded.
    fn store(
        &mut self,
        id: AssetId<Shader>,
        shader_defs: &[ShaderDefVal],
        module: Arc<ShaderModule>,
        wesl_dependencies: Vec<AssetId<Shader>>,
    ) -> Arc<ShaderModule> {
        let data = self.data.entry(id).or_default();
        let module = match data.processed_shaders.entry_ref(shader_defs) {
            EntryRef::Occupied(entry) => entry.get().clone(),
            EntryRef::Vacant(entry) => entry.insert_with_key(shader_defs.into(), module).clone(),
        };
        self.add_dependents(id, &wesl_dependencies);
        module
    }

    fn add_dependents(&mut self, id: AssetId<Shader>, wesl_dependencies: &[AssetId<Shader>]) {
        for dep_id in wesl_dependencies {
            self.data.entry(*dep_id).or_default().dependents.insert(id);
        }
    }

    fn clear(&mut self, id: AssetId<Shader>) -> Vec<CachedPipelineId> {
        let mut shaders_to_clear = vec![id];
        let mut visited = HashSet::new();
        let mut pipelines_to_queue = Vec::new();
        while let Some(handle) = shaders_to_clear.pop() {
            if !visited.insert(handle) {
                continue;
            }
            let changes = self.changes.entry(handle).or_default();
            *changes = changes.wrapping_add(1);
            if let Some(data) = self.data.get_mut(&handle) {
                data.processed_shaders.clear();
                pipelines_to_queue.extend(data.pipelines.iter().copied());
                shaders_to_clear.extend(data.dependents.iter().copied());
            }
        }

        pipelines_to_queue
    }

    /// Inserts and possibly replaces a shader at the given asset id.
    ///
    /// Returns a vec of which cached pipelines depended on it
    /// (directly or indirectly via a shader import) and thus must be recompiled.
    pub fn set_shader(&mut self, id: AssetId<Shader>, shader: Shader) -> Vec<CachedPipelineId> {
        self.missing_import_logged.clear();
        let mut pipelines_to_queue = self.clear(id);
        let path = &shader.import_path;
        self.import_path_shaders.insert(path.clone(), id);
        if let Some(waiting_shaders) = self.waiting_on_import.get_mut(path) {
            for waiting_shader in waiting_shaders.drain(..) {
                // resolve waiting shader import
                let data = self.data.entry(waiting_shader).or_default();
                data.resolved_imports.insert(path.clone(), id);
                // add waiting shader as dependent of this shader
                let data = self.data.entry(id).or_default();
                data.dependents.insert(waiting_shader);
            }
        }

        for import in shader.imports.iter() {
            if let Some(import_id) = self.import_path_shaders.get(import).copied() {
                // resolve import because it is currently available
                let data = self.data.entry(id).or_default();
                data.resolved_imports.insert(import.clone(), import_id);
                // add this shader as a dependent of the import
                let data = self.data.entry(import_id).or_default();
                data.dependents.insert(id);
            } else {
                let waiting = self.waiting_on_import.entry(import.clone()).or_default();
                waiting.push(id);
            }
        }

        if let Source::Wesl(_) = shader.source {
            match wesl_module_path(&shader.import_path) {
                Some(module_path) => {
                    if let Some(existing) = self.module_path_to_asset_id.get(&module_path).copied()
                        && existing != id
                    {
                        warn!(
                            "Shader module path `{module_path}` was registered to a different \
                            shader, replacing with `{}`",
                            shader.path
                        );
                        pipelines_to_queue.extend(self.clear(existing));
                    }
                    self.module_path_to_asset_id.insert(module_path, id);
                }
                None => warn!(
                    "Shader `{}` has a malformed import path `{:?}`",
                    shader.path, shader.import_path
                ),
            }
        }
        self.shaders.insert(id, Arc::new(shader));
        pipelines_to_queue
    }

    /// Removes the shader with the given asset id.
    ///
    /// Returns a vec of which cached pipelines depended on it
    /// (directly or indirectly via a shader import) and thus must be recompiled.
    pub fn remove(&mut self, id: AssetId<Shader>) -> Vec<CachedPipelineId> {
        let pipelines_to_queue = self.clear(id);
        if let Some(shader) = self.shaders.remove(&id) {
            if self.import_path_shaders.get(&shader.import_path) == Some(&id) {
                self.import_path_shaders.remove(&shader.import_path);
            }
            if let Source::Wesl(_) = shader.source
                && let Some(module_path) = wesl_module_path(&shader.import_path)
                && self.module_path_to_asset_id.get(&module_path) == Some(&id)
            {
                self.module_path_to_asset_id.remove(&module_path);
            }
        }

        pipelines_to_queue
    }
}

/// A shader's source made ready for `load_module`: the composed WGSL, or the shader's own SPIR-V.
enum Composed {
    Wgsl(String),
    SpirV,
}

impl Composed {
    fn into_source(self, shader: &Shader) -> ShaderCacheSource<'_> {
        match (self, &shader.source) {
            (Composed::Wgsl(s), _) => ShaderCacheSource::Wgsl(s),
            (Composed::SpirV, Source::SpirV(data)) => ShaderCacheSource::SpirV(data.as_ref()),
            (Composed::SpirV, _) => unreachable!("composed as SPIR-V from a text source"),
        }
    }
}

enum ComposeError {
    ImportNotFound(String),
    Other(ShaderCacheError),
}

enum Prepared<ShaderModule> {
    Cached(Arc<ShaderModule>),
    Compose(Arc<Shader>),
}

/// Composes one variant of a shader. Called by [`ShaderCache::get`] under the cache's lock, and by
/// [`ShaderCache::get_shared`] outside it, on a snapshot of the cache's shaders.
fn compose(
    shaders: &HashMap<AssetId<Shader>, Arc<Shader>>,
    import_path_shaders: &HashMap<ShaderImport, AssetId<Shader>>,
    module_path_to_asset_id: &HashMap<wesl::syntax::ModulePath, AssetId<Shader>>,
    id: AssetId<Shader>,
    shader: &Shader,
    shader_defs: &[ShaderDefVal],
) -> Result<(Composed, Vec<AssetId<Shader>>), ComposeError> {
    let mut wesl_dependencies: Vec<AssetId<Shader>> = Vec::new();
    match &shader.source {
        Source::SpirV(_) => Ok((Composed::SpirV, wesl_dependencies)),
        Source::Wgsl(wgsl_source) => {
            Ok((Composed::Wgsl(wgsl_source.to_string()), wesl_dependencies))
        }
        Source::Wesl(_) => {
            if let Some(module_path) = wesl_module_path(&shader.import_path) {
                let mut compiler_options = wesl::CompileOptions {
                    imports: true,
                    condcomp: true,
                    visibility: false,
                    ..Default::default()
                };

                let mut closure_defs = Vec::new();
                let mut visited = HashSet::new();
                let mut to_visit = vec![id];
                while let Some(dep_id) = to_visit.pop() {
                    if !visited.insert(dep_id) {
                        continue;
                    }
                    let Some(dep) = shaders.get(&dep_id) else {
                        continue;
                    };
                    if dep_id != id {
                        closure_defs.extend(dep.shader_defs.iter());
                    }
                    for import in &dep.imports {
                        if let Some(import_id) = import_path_shaders.get(import) {
                            to_visit.push(*import_id);
                        }
                    }
                }
                let mut constants = alloc::collections::BTreeMap::new();
                for shader_def in closure_defs
                    .into_iter()
                    .chain(shader_defs.iter())
                    .chain(shader.shader_defs.iter())
                {
                    match shader_def {
                        ShaderDefVal::Bool(key, value) => {
                            compiler_options
                                .features
                                .flags
                                .insert(key.to_string(), (*value).into());
                        }
                        ShaderDefVal::Int(key, value) => {
                            compiler_options
                                .features
                                .flags
                                .insert(key.to_string(), true.into());
                            constants.insert(key.as_ref(), value.to_string());
                        }
                        ShaderDefVal::UInt(key, value) => {
                            compiler_options
                                .features
                                .flags
                                .insert(key.to_string(), true.into());
                            constants.insert(key.as_ref(), value.to_string());
                        }
                    }
                }
                let constants_source: String = constants
                    .iter()
                    .map(|(name, value)| format!("const {name} = {value};\n"))
                    .collect();

                let shader_resolver =
                    ShaderResolver::new(module_path_to_asset_id, shaders, &constants_source);

                let compiled = wesl::compile(&module_path, &compiler_options, &shader_resolver)
                    .map_err(|error| {
                        // We use render_plain to avoid rendering ANSI codes
                        // Workaround for: https://github.com/tokio-rs/tracing/issues/3378
                        if is_module_not_found(&error) {
                            ComposeError::ImportNotFound(error.diagnostic().render_plain())
                        } else {
                            ComposeError::Other(ShaderCacheError::ProcessShaderError(
                                error.diagnostic().render_plain(),
                            ))
                        }
                    })?;

                for used in &compiled.modules {
                    let used_path = match &used.path.origin {
                        wesl::syntax::PathOrigin::Package(pkg) if pkg.contains('/') => {
                            Cow::Owned(wesl::syntax::ModulePath {
                                origin: wesl::syntax::PathOrigin::Package(
                                    pkg.rsplit('/').next().unwrap().to_string(),
                                ),
                                components: used.path.components.clone(),
                            })
                        }
                        _ => Cow::Borrowed(&used.path),
                    };
                    if let Some(dep_id) = module_path_to_asset_id.get(used_path.as_ref())
                        && *dep_id != id
                    {
                        wesl_dependencies.push(*dep_id);
                    }
                }

                Ok((Composed::Wgsl(compiled.to_string()), wesl_dependencies))
            } else {
                Err(ComposeError::Other(ShaderCacheError::ProcessShaderError(
                    format!(
                        "Wesl shader `{}` has a malformed import path `{:?}`",
                        shader.path, shader.import_path
                    ),
                )))
            }
        }
    }
}

/// A Wesl import resolver. Maps module paths to actual Wesl shader source.
pub struct ShaderResolver<'a> {
    module_path_to_asset_id: &'a HashMap<wesl::syntax::ModulePath, AssetId<Shader>>,
    shaders: &'a HashMap<AssetId<Shader>, Arc<Shader>>,
    constants_source: &'a str,
}

impl<'a> ShaderResolver<'a> {
    /// Creates a shader resolver with the given map of module paths to shader asset ids,
    /// map of shader asset ids to shader source, and the source of the virtual
    /// `constants` module. This resolver is not meant to be long living.
    pub fn new(
        module_path_to_asset_id: &'a HashMap<wesl::syntax::ModulePath, AssetId<Shader>>,
        shaders: &'a HashMap<AssetId<Shader>, Arc<Shader>>,
        constants_source: &'a str,
    ) -> Self {
        Self {
            module_path_to_asset_id,
            shaders,
            constants_source,
        }
    }
}

impl<'a> wesl::Resolver for ShaderResolver<'a> {
    fn resolve_source(
        &self,
        module_path: &wesl::syntax::ModulePath,
    ) -> Result<Cow<'_, str>, wesl::error::ResolveError> {
        let module_path = self.canonical_path(module_path);
        if module_path.origin == wesl::syntax::PathOrigin::Package("constants".to_string())
            && module_path.components.is_empty()
        {
            return Ok(Cow::Borrowed(self.constants_source));
        }
        let asset_id = self
            .module_path_to_asset_id
            .get(&module_path)
            .ok_or_else(|| {
                wesl::error::ResolveError::ModuleNotFound(
                    module_path.clone(),
                    "no shader registered for this module path".to_string(),
                )
            })?;

        let shader = self.shaders.get(asset_id).ok_or_else(|| {
            wesl::error::ResolveError::ModuleNotFound(
                module_path.clone(),
                "shader asset not loaded".to_string(),
            )
        })?;
        Ok(Cow::Borrowed(shader.source.as_str()))
    }

    fn canonical_path(&self, module_path: &wesl::syntax::ModulePath) -> wesl::syntax::ModulePath {
        match &module_path.origin {
            wesl::syntax::PathOrigin::Package(pkg) if pkg.contains('/') => {
                wesl::syntax::ModulePath {
                    origin: wesl::syntax::PathOrigin::Package(
                        pkg.rsplit('/').next().unwrap().to_string(),
                    ),
                    components: module_path.components.clone(),
                }
            }
            _ => module_path.clone(),
        }
    }

    fn display_name(&self, module_path: &wesl::syntax::ModulePath) -> Option<String> {
        let module_path = self.canonical_path(module_path);
        let asset_id = self.module_path_to_asset_id.get(&module_path)?;
        let shader = self.shaders.get(asset_id)?;
        Some(shader.path.clone())
    }
}

/// Type of error returned by a `PipelineCache` when the creation of a GPU pipeline object failed.
#[expect(missing_docs, reason = "Enum variants are self-explanatory")]
#[derive(Error, Debug)]
pub enum ShaderCacheError {
    #[error(
        "Pipeline could not be compiled because the following shader could not be loaded: {0:?}"
    )]
    ShaderNotLoaded(AssetId<Shader>),
    #[error("Failed to process shader:\n{0}")]
    ProcessShaderError(String),
    #[error("Shader import not yet available.")]
    ShaderImportNotYetAvailable,
    #[error("Could not create shader module: {0}")]
    CreateShaderModule(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cache() -> ShaderCache<String, ()> {
        ShaderCache::new((), |_, source, _| match source {
            ShaderCacheSource::Wgsl(wgsl) => Ok(wgsl),
            _ => panic!("expected wgsl output"),
        })
    }

    #[test]
    fn import_resolution() {
        let mut cache = test_cache();

        let (maths_id, lighting_id, root_id) = set_test_shaders(&mut cache);

        let compiled = cache
            .get(0, root_id, &[ShaderDefVal::Bool("BRIGHT".into(), true)])
            .unwrap();
        assert!(compiled.contains("fn fragment"));
        assert!(compiled.contains("* 2.0"));
        assert!(compiled.contains("+ 0.1"));

        let (maths, lighting, _) = test_shaders();
        assert!(cache.set_shader(lighting_id, lighting).contains(&0));
        let _ = cache.get(0, root_id, &[ShaderDefVal::Bool("BRIGHT".into(), true)]);
        assert!(cache.set_shader(maths_id, maths).contains(&0));
    }

    #[test]
    fn constants_module() {
        let mut cache = test_cache();
        let mut shader = Shader::from_wesl(
            r#"
@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> scale: f32;
var<uniform> batch: array<vec4<f32>, constants::BATCH_SIZE>;

@fragment
fn fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(scale) + batch[0];
}
"#,
            "shaders/constants_user.wesl",
        );
        shader.shader_defs = vec![ShaderDefVal::UInt("BATCH_SIZE".into(), 4)];
        let id = AssetId::Uuid {
            uuid: bevy_asset::uuid::Uuid::from_u128(7),
        };
        cache.set_shader(id, shader);

        let compiled = cache
            .get(
                0,
                id,
                &[ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 2)],
            )
            .unwrap();
        assert!(compiled.contains("= 2;"));
        assert!(compiled.contains("= 4;"));
    }

    #[test]
    fn compile_error_display_names() {
        let mut cache = test_cache();
        let (maths, _, root) = test_shaders();
        let (maths_id, lighting_id, root_id) = test_ids();
        let broken_lighting = Shader::from_wesl(
            "fn brighten(x: f32) -> f32 { return x + ; }",
            "embedded://bevy_pbr/render/lighting.wesl",
        );
        cache.set_shader(maths_id, maths);
        cache.set_shader(lighting_id, broken_lighting);
        cache.set_shader(root_id, root);

        let error = cache
            .get(0, root_id, &[ShaderDefVal::Bool("BRIGHT".into(), true)])
            .expect_err("syntax error");
        let ShaderCacheError::ProcessShaderError(message) = error else {
            panic!("expected ProcessShaderError, got: {error:?}");
        };
        assert!(message.contains("embedded://bevy_pbr/render/lighting.wesl"));
    }

    #[test]
    fn import_retry() {
        let mut cache = test_cache();
        let (maths, lighting, root) = test_shaders();
        let (maths_id, lighting_id, root_id) = test_ids();

        cache.set_shader(root_id, root);
        assert!(matches!(
            cache.get(0, root_id, &[]),
            Err(ShaderCacheError::ShaderImportNotYetAvailable)
        ));

        cache.set_shader(lighting_id, lighting);
        cache.set_shader(maths_id, maths);
        cache.get(0, root_id, &[]).unwrap();
    }

    #[test]
    fn get_shared_recomposes_after_a_concurrent_change() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Mutex, OnceLock,
        };

        // `load_module` runs outside the cache's lock, so it stands in for another thread that
        // hot-reloads an import while this variant is being composed.
        static CACHE: OnceLock<Mutex<ShaderCache<String, ()>>> = OnceLock::new();
        static RELOAD: AtomicBool = AtomicBool::new(true);
        let cache = CACHE.get_or_init(|| {
            Mutex::new(ShaderCache::new((), |_, source, _| {
                let ShaderCacheSource::Wgsl(wgsl) = source else {
                    panic!("expected wgsl output");
                };
                if RELOAD.swap(false, Ordering::SeqCst) {
                    let (_, lighting_id, _) = test_ids();
                    let lighting = Shader::from_wesl(
                        "fn brighten(x: f32) -> f32 { return x + 0.5; }",
                        "embedded://bevy_pbr/render/lighting.wesl",
                    );
                    CACHE
                        .get()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .set_shader(lighting_id, lighting);
                }
                Ok(wgsl)
            }))
        });
        let (_, _, root_id) = set_test_shaders(&mut cache.lock().unwrap());
        let create = Mutex::new(());
        let defs = [ShaderDefVal::Bool("BRIGHT".into(), true)];

        // As with `get`, the first call may hand back the variant made from the old import, but
        // it must not be cached: the reload queued the pipeline again, and that call sees it.
        let _ = ShaderCache::get_shared(cache, &create, 0, root_id, &defs).unwrap();
        assert!(!RELOAD.load(Ordering::SeqCst));
        let compiled = ShaderCache::get_shared(cache, &create, 0, root_id, &defs).unwrap();
        assert!(compiled.contains("+ 0.5"));
        assert!(!compiled.contains("+ 0.1"));

        let cached = ShaderCache::get_shared(cache, &create, 0, root_id, &defs).unwrap();
        assert!(Arc::ptr_eq(&compiled, &cached));
    }

    #[test]
    fn get_shared_recomposes_after_a_dependency_changes() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Mutex, OnceLock,
        };

        // `maths` is used through a qualified path, not an import, so `clear` cannot reach the root
        // through it until a variant is stored: `get_shared` must see the change and compose again.
        static CACHE: OnceLock<Mutex<ShaderCache<String, ()>>> = OnceLock::new();
        static RELOAD: AtomicBool = AtomicBool::new(true);
        let cache = CACHE.get_or_init(|| {
            Mutex::new(ShaderCache::new((), |_, source, _| {
                let ShaderCacheSource::Wgsl(wgsl) = source else {
                    panic!("expected wgsl output");
                };
                if RELOAD.swap(false, Ordering::SeqCst) {
                    let (maths_id, _, _) = test_ids();
                    let maths = Shader::from_wesl(
                        "fn double(x: f32) -> f32 { return x * 3.0; }",
                        "embedded://bevy_render/maths.wesl",
                    );
                    CACHE
                        .get()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .set_shader(maths_id, maths);
                }
                Ok(wgsl)
            }))
        });
        let (_, _, root_id) = set_test_shaders(&mut cache.lock().unwrap());
        let create = Mutex::new(());
        let defs = [ShaderDefVal::Bool("BRIGHT".into(), true)];

        let compiled = ShaderCache::get_shared(cache, &create, 0, root_id, &defs).unwrap();
        assert!(!RELOAD.load(Ordering::SeqCst));
        assert!(compiled.contains("* 3.0"));
        assert!(!compiled.contains("* 2.0"));

        let cached = ShaderCache::get_shared(cache, &create, 0, root_id, &defs).unwrap();
        assert!(Arc::ptr_eq(&compiled, &cached));
    }

    fn test_shaders() -> (Shader, Shader, Shader) {
        let maths = Shader::from_wesl(
            "fn double(x: f32) -> f32 { return x * 2.0; }",
            "embedded://bevy_render/maths.wesl",
        );
        let lighting = Shader::from_wesl(
            "fn brighten(x: f32) -> f32 { return x + 0.1; }",
            "embedded://bevy_pbr/render/lighting.wesl",
        );
        let root = Shader::from_wesl(
            r#"
import bevy_pbr::render::lighting::brighten;

@fragment
fn fragment() -> @location(0) vec4<f32> {
    var value = bevy_render::maths::double(1.0);
    @if(BRIGHT) { value = brighten(value); }
    return vec4<f32>(value);
}
"#,
            "shaders/root.wesl",
        );
        (maths, lighting, root)
    }

    fn test_ids() -> (AssetId<Shader>, AssetId<Shader>, AssetId<Shader>) {
        let id = |n| AssetId::Uuid {
            uuid: bevy_asset::uuid::Uuid::from_u128(n),
        };
        (id(1), id(2), id(3))
    }

    #[test]
    fn library_def_scoping() {
        let mut cache = test_cache();
        let id = |n| AssetId::Uuid {
            uuid: bevy_asset::uuid::Uuid::from_u128(n),
        };

        let mut lib_a = Shader::from_wesl(
            "var<uniform> batch_a: array<vec4<f32>, constants::BATCH_SIZE>;",
            "embedded://bevy_a/bindings.wesl",
        );
        lib_a.shader_defs = vec![ShaderDefVal::UInt("BATCH_SIZE".into(), 3)];
        let mut lib_b = Shader::from_wesl(
            "var<uniform> batch_b: array<vec4<f32>, constants::BATCH_SIZE>;",
            "embedded://bevy_b/bindings.wesl",
        );
        lib_b.shader_defs = vec![ShaderDefVal::UInt("BATCH_SIZE".into(), 7)];

        let root_a = Shader::from_wesl(
            r#"
import bevy_a::bindings::batch_a;
@fragment
fn fragment() -> @location(0) vec4<f32> { return batch_a[0]; }
"#,
            "shaders/root_a.wesl",
        );
        let root_b = Shader::from_wesl(
            r#"
import bevy_b::bindings::batch_b;
@fragment
fn fragment() -> @location(0) vec4<f32> { return batch_b[0]; }
"#,
            "shaders/root_b.wesl",
        );

        cache.set_shader(id(1), lib_a);
        cache.set_shader(id(2), lib_b);
        cache.set_shader(id(3), root_a);
        cache.set_shader(id(4), root_b);

        let compiled_a = cache.get(0, id(3), &[]).unwrap();
        let compiled_b = cache.get(1, id(4), &[]).unwrap();
        assert!(compiled_a.contains("= 3;") && !compiled_a.contains("= 7;"));
        assert!(compiled_b.contains("= 7;") && !compiled_b.contains("= 3;"));
    }

    #[test]
    fn cyclic_import_invalidation() {
        let mut cache = test_cache();
        let id = |n| AssetId::Uuid {
            uuid: bevy_asset::uuid::Uuid::from_u128(n),
        };
        let module_a = Shader::from_wesl(
            "import bevy_cycle::b::from_b;\nfn from_a() -> f32 { return 1.0; }",
            "embedded://bevy_cycle/a.wesl",
        );
        let module_b = Shader::from_wesl(
            "import bevy_cycle::a::from_a;\nfn from_b() -> f32 { return 2.0; }",
            "embedded://bevy_cycle/b.wesl",
        );
        cache.set_shader(id(1), module_a);
        cache.set_shader(id(2), module_b);
        let module_a = Shader::from_wesl(
            "import bevy_cycle::b::from_b;\nfn from_a() -> f32 { return 3.0; }",
            "embedded://bevy_cycle/a.wesl",
        );
        cache.set_shader(id(1), module_a);
    }

    fn set_test_shaders(
        cache: &mut ShaderCache<String, ()>,
    ) -> (AssetId<Shader>, AssetId<Shader>, AssetId<Shader>) {
        let (maths, lighting, root) = test_shaders();
        let (maths_id, lighting_id, root_id) = test_ids();
        cache.set_shader(maths_id, maths);
        cache.set_shader(lighting_id, lighting);
        cache.set_shader(root_id, root);
        (maths_id, lighting_id, root_id)
    }
}
