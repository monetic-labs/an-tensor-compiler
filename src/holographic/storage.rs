//! Tensor storage and persistence
//!
//! Save and load holographic tensors using safetensors format.
//!
//! ## Features
//!
//! - **Organism storage**: Full organism tensors with contexts and metadata
//! - **Component caching**: Individual component tensors by content hash
//! - **Incremental updates**: Git-based change detection for partial re-encoding

use crate::holographic::types::*;
use crate::{Result, TensorCoreError, Device, Tensor};
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;
use sha2::{Sha256, Digest};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};

// ============================================================================
// Tensor Store
// ============================================================================

/// Store for persisting and loading organism tensors
pub struct TensorStore {
    /// Base directory for tensor storage
    base_path: std::path::PathBuf,

    /// Device for loaded tensors
    device: Device,
}

impl TensorStore {
    /// Create a new tensor store at the given base path
    pub fn new(base_path: impl AsRef<Path>, device: Device) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            device,
        }
    }
    
    /// Save an organism tensor to disk
    pub fn save_organism(&self, organism: &OrganismTensor) -> Result<()> {
        let dir = self.base_path.join(&organism.project_id);
        std::fs::create_dir_all(&dir)
            .map_err(TensorCoreError::Io)?;
        
        // Save main organism hologram
        let organism_path = dir.join("organism.safetensors");
        self.save_tensor(&organism.hologram, &organism_path)?;
        
        // Save metadata
        let metadata = OrganismMetadata {
            project_id: organism.project_id.clone(),
            name: organism.name.clone(),
            description: organism.description.clone(),
            commit: organism.commit.clone(),
            updated_at: organism.updated_at,
            embedding_dim: organism.embedding_dim,
            context_names: organism.contexts.iter().map(|c| c.name.clone()).collect(),
            federation: organism.federation.clone(),
        };
        let metadata_path = dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        std::fs::write(&metadata_path, metadata_json)
            .map_err(TensorCoreError::Io)?;
        
        // Save boundaries
        let boundaries_path = dir.join("boundaries.json");
        let boundaries_json = serde_json::to_string_pretty(&organism.boundaries)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        std::fs::write(&boundaries_path, boundaries_json)
            .map_err(TensorCoreError::Io)?;
        
        // Save each context
        for ctx in &organism.contexts {
            self.save_context(ctx, &dir)?;
        }
        
        Ok(())
    }
    
    /// Save a bounded context
    fn save_context(&self, ctx: &BoundedContextTensor, base_dir: &Path) -> Result<()> {
        let ctx_dir = base_dir.join("contexts").join(&ctx.name);
        std::fs::create_dir_all(&ctx_dir)
            .map_err(TensorCoreError::Io)?;
        
        // Save context hologram
        let hologram_path = ctx_dir.join("hologram.safetensors");
        self.save_tensor(&ctx.hologram, &hologram_path)?;
        
        // Save context metadata
        let ctx_meta = ContextMetadata {
            name: ctx.name.clone(),
            description: ctx.description.clone(),
            role: format!("{:?}", ctx.role),
            patterns: ctx.patterns.clone(),
            api: ctx.api.iter().map(|c| c.full_path()).collect(),
            module_count: ctx.modules.len(),
            component_count: ctx.modules.iter().map(|m| m.components.len()).sum(),
        };
        let meta_path = ctx_dir.join("metadata.json");
        let meta_json = serde_json::to_string_pretty(&ctx_meta)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        std::fs::write(&meta_path, meta_json)
            .map_err(TensorCoreError::Io)?;
        
        // Save module holograms (batched into one file)
        let mut module_tensors = HashMap::new();
        for (i, module) in ctx.modules.iter().enumerate() {
            let key = format!("module_{}", i);
            module_tensors.insert(key, &module.hologram);
        }
        
        if !module_tensors.is_empty() {
            let modules_path = ctx_dir.join("modules.safetensors");
            self.save_tensors(&module_tensors, &modules_path)?;
        }
        
        // Save component holograms (batched into one file per module to avoid huge files)
        for (mod_idx, module) in ctx.modules.iter().enumerate() {
            if module.components.is_empty() {
                continue;
            }
            
            let mut comp_tensors = HashMap::new();
            for (comp_idx, comp) in module.components.iter().enumerate() {
                let key = format!("comp_{}", comp_idx);
                comp_tensors.insert(key, &comp.hologram);
            }
            
            let comp_path = ctx_dir.join(format!("components_{}.safetensors", mod_idx));
            self.save_tensors(&comp_tensors, &comp_path)?;
        }
        
        // Save component index (for fast lookup and reconstruction)
        let component_index: Vec<ComponentIndexEntry> = ctx.modules.iter()
            .enumerate()
            .flat_map(|(mod_idx, m)| m.components.iter().enumerate().map(move |(comp_idx, c)| {
                ComponentIndexEntry {
                    id: c.id.full_path(),
                    kind: format!("{:?}", c.id.kind),
                    file: c.file.to_string_lossy().to_string(),
                    line_start: c.line_start,
                    line_end: c.line_end,
                    is_public: c.is_public,
                    module_idx: mod_idx,
                    component_idx: comp_idx,
                }
            }))
            .collect();
        
        let index_path = ctx_dir.join("components.json");
        let index_json = serde_json::to_string_pretty(&component_index)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        std::fs::write(&index_path, index_json)
            .map_err(TensorCoreError::Io)?;
        
        Ok(())
    }
    
    /// Load an organism tensor from disk
    pub fn load_organism(&self, project_id: &str) -> Result<OrganismTensor> {
        let dir = self.base_path.join(project_id);
        
        if !dir.exists() {
            return Err(TensorCoreError::Config(
                format!("Organism not found: {}", project_id)
            ));
        }
        
        // Load metadata
        let metadata_path = dir.join("metadata.json");
        let metadata_json = std::fs::read_to_string(&metadata_path)
            .map_err(TensorCoreError::Io)?;
        let metadata: OrganismMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        
        // Load boundaries
        let boundaries_path = dir.join("boundaries.json");
        let boundaries: BoundaryRules = if boundaries_path.exists() {
            let json = std::fs::read_to_string(&boundaries_path)
                .map_err(TensorCoreError::Io)?;
            serde_json::from_str(&json)
                .map_err(|e| TensorCoreError::Serialization(e.to_string()))?
        } else {
            BoundaryRules::default()
        };
        
        // Load organism hologram
        let organism_path = dir.join("organism.safetensors");
        let hologram = self.load_tensor(&organism_path)?;
        
        // Load contexts
        let mut contexts = Vec::new();
        for ctx_name in &metadata.context_names {
            if let Ok(ctx) = self.load_context(ctx_name, &dir) {
                contexts.push(ctx);
            }
        }
        
        Ok(OrganismTensor {
            project_id: metadata.project_id,
            name: metadata.name,
            description: metadata.description,
            contexts,
            hologram,
            boundaries,
            federation: metadata.federation,
            history: None,
            updated_at: metadata.updated_at,
            commit: metadata.commit,
            embedding_dim: metadata.embedding_dim,
        })
    }
    
    /// Load a bounded context with all modules and components
    fn load_context(&self, name: &str, base_dir: &Path) -> Result<BoundedContextTensor> {
        let ctx_dir = base_dir.join("contexts").join(name);
        
        // Load metadata
        let meta_path = ctx_dir.join("metadata.json");
        let meta_json = std::fs::read_to_string(&meta_path)
            .map_err(TensorCoreError::Io)?;
        let meta: ContextMetadata = serde_json::from_str(&meta_json)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        
        // Load hologram
        let hologram_path = ctx_dir.join("hologram.safetensors");
        let hologram = self.load_tensor(&hologram_path)?;
        
        // Parse role
        let role = ContextRole::parse(&meta.role).unwrap_or(ContextRole::Utility);
        
        // Load component index
        let index_path = ctx_dir.join("components.json");
        let component_index: Vec<ComponentIndexEntry> = if index_path.exists() {
            let json = std::fs::read_to_string(&index_path)
                .map_err(TensorCoreError::Io)?;
            serde_json::from_str(&json)
                .map_err(|e| TensorCoreError::Serialization(e.to_string()))?
        } else {
            Vec::new()
        };
        
        // Load module holograms
        let modules_path = ctx_dir.join("modules.safetensors");
        let module_holograms = if modules_path.exists() {
            self.load_named_tensors(&modules_path)?
        } else {
            HashMap::new()
        };
        
        // Group components by module
        let mut components_by_module: HashMap<usize, Vec<&ComponentIndexEntry>> = HashMap::new();
        for entry in &component_index {
            components_by_module.entry(entry.module_idx).or_default().push(entry);
        }
        
        // Build modules with components
        let mut modules = Vec::new();
        for mod_idx in 0..meta.module_count {
            let mod_key = format!("module_{}", mod_idx);
            let module_hologram = module_holograms.get(&mod_key)
                .cloned()
                .unwrap_or_else(|| Tensor::zeros((1,), candle_core::DType::F32, &self.device).unwrap());
            
            // Load component holograms for this module
            let comp_path = ctx_dir.join(format!("components_{}.safetensors", mod_idx));
            let comp_holograms = if comp_path.exists() {
                self.load_named_tensors(&comp_path)?
            } else {
                HashMap::new()
            };
            
            // Build components
            let mut components = Vec::new();
            if let Some(comp_entries) = components_by_module.get(&mod_idx) {
                for entry in comp_entries {
                    let comp_key = format!("comp_{}", entry.component_idx);
                    let comp_hologram = comp_holograms.get(&comp_key)
                        .cloned()
                        .unwrap_or_else(|| Tensor::zeros((1,), candle_core::DType::F32, &self.device).unwrap());
                    
                    // Parse component kind
                    let kind = match entry.kind.as_str() {
                        "Struct" => ComponentKind::Struct,
                        "Enum" => ComponentKind::Enum,
                        "Trait" => ComponentKind::Trait,
                        "Function" => ComponentKind::Function,
                        "Impl" => ComponentKind::Impl,
                        "Const" => ComponentKind::Const,
                        "Static" => ComponentKind::Static,
                        "TypeAlias" => ComponentKind::TypeAlias,
                        "Macro" => ComponentKind::Macro,
                        "Module" => ComponentKind::Module,
                        _ => ComponentKind::Function, // Fallback
                    };
                    
                    // Parse component ID from full path
                    let parts: Vec<&str> = entry.id.split("::").collect();
                    let comp_name = parts.last().unwrap_or(&"unknown").to_string();
                    let module_path = if parts.len() > 1 {
                        parts[..parts.len()-1].join("::")
                    } else {
                        name.to_string()
                    };
                    
                    // Use the hologram as semantic, or create zero tensors for others
                    let dim = comp_hologram.dims().first().copied().unwrap_or(1024);
                    let zero_tensor = Tensor::zeros((dim,), candle_core::DType::F32, &self.device)
                        .unwrap_or(comp_hologram.clone());
                    
                    components.push(ComponentTensor {
                        id: ComponentId::new(&module_path, &comp_name, kind),
                        file: PathBuf::from(&entry.file),
                        line_start: entry.line_start,
                        line_end: entry.line_end,
                        semantic: comp_hologram.clone(),
                        structural: zero_tensor.clone(),
                        signature: zero_tensor.clone(),
                        documentation: None,
                        hologram: comp_hologram,
                        is_public: entry.is_public,
                    });
                }
            }
            
            // Get module path from first component or use generic name
            let module_path = components.first()
                .map(|c| c.id.module.clone())
                .unwrap_or_else(|| format!("{}::module_{}", name, mod_idx));
            
            // Get public exports from components
            let exports: Vec<ComponentId> = components.iter()
                .filter(|c| c.is_public)
                .map(|c| c.id.clone())
                .collect();
            
            modules.push(ModuleTensor {
                path: module_path,
                file: PathBuf::new(),
                components,
                hologram: module_hologram,
                exports,
                imports: Vec::new(),
            });
        }
        
        // Extract API items from public components
        let api: Vec<ComponentId> = modules.iter()
            .flat_map(|m| m.exports.iter().cloned())
            .collect();
        
        Ok(BoundedContextTensor {
            name: meta.name,
            path: ctx_dir,
            description: meta.description,
            role,
            modules,
            hologram,
            api,
            patterns: meta.patterns,
            dependencies: Vec::new(),
        })
    }
    
    /// Load multiple named tensors from a safetensors file
    fn load_named_tensors(&self, path: &Path) -> Result<HashMap<String, Tensor>> {
        let data = std::fs::read(path)
            .map_err(TensorCoreError::Io)?;
        
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        
        let mut result = HashMap::new();
        for name in tensors.names() {
            let view = tensors.tensor(name)
                .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
            
            let shape: Vec<usize> = view.shape().to_vec();
            let float_data: Vec<f32> = bytemuck::cast_slice(view.data()).to_vec();
            
            let tensor = Tensor::from_vec(float_data, shape.as_slice(), &self.device)
                .map_err(|e| TensorCoreError::Tensor(e.to_string()))?;
            
            result.insert(name.to_string(), tensor);
        }
        
        Ok(result)
    }
    
    /// Save a single tensor
    fn save_tensor(&self, tensor: &Tensor, path: &Path) -> Result<()> {
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        let shape: Vec<usize> = tensor.dims().to_vec();
        
        let tensor_data = safetensors::tensor::Dtype::F32;
        let view = TensorView::new(
            tensor_data,
            shape,
            bytemuck::cast_slice(&data),
        ).map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        
        let tensors: HashMap<String, TensorView> = [("tensor".to_string(), view)].into();
        
        safetensors::serialize_to_file(tensors, &None, path)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        
        Ok(())
    }
    
    /// Save multiple tensors
    fn save_tensors(&self, tensors: &HashMap<String, &Tensor>, path: &Path) -> Result<()> {
        let mut views: HashMap<String, TensorView> = HashMap::new();
        let mut data_store: HashMap<String, Vec<f32>> = HashMap::new();
        
        for (name, tensor) in tensors {
            let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
            data_store.insert(name.clone(), data);
        }
        
        for (name, tensor) in tensors {
            let data = data_store.get(name).unwrap();
            let shape: Vec<usize> = tensor.dims().to_vec();
            
            let view = TensorView::new(
                safetensors::tensor::Dtype::F32,
                shape,
                bytemuck::cast_slice(data),
            ).map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
            
            views.insert(name.clone(), view);
        }
        
        safetensors::serialize_to_file(views, &None, path)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        
        Ok(())
    }
    
    /// Load a single tensor
    fn load_tensor(&self, path: &Path) -> Result<Tensor> {
        let data = std::fs::read(path)
            .map_err(TensorCoreError::Io)?;
        
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        
        let view = tensors.tensor("tensor")
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        
        let shape: Vec<usize> = view.shape().to_vec();
        let float_data: Vec<f32> = bytemuck::cast_slice(view.data()).to_vec();
        
        Tensor::from_vec(float_data, shape.as_slice(), &self.device)
            .map_err(|e| TensorCoreError::Tensor(e.to_string()))
    }
    
    /// Check if an organism is stored
    pub fn exists(&self, project_id: &str) -> bool {
        self.base_path.join(project_id).join("organism.safetensors").exists()
    }
    
    /// List stored organisms
    pub fn list_organisms(&self) -> Result<Vec<String>> {
        let mut organisms = Vec::new();
        
        if !self.base_path.exists() {
            return Ok(organisms);
        }
        
        for entry in std::fs::read_dir(&self.base_path)
            .map_err(TensorCoreError::Io)?
        {
            let entry = entry.map_err(TensorCoreError::Io)?;
            let path = entry.path();
            
            if path.is_dir() && path.join("organism.safetensors").exists() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    organisms.push(name.to_string());
                }
            }
        }
        
        Ok(organisms)
    }
    
    /// Get the cache for this store
    pub fn cache(&self) -> ComponentCache {
        ComponentCache::new(self.base_path.join("cache"), self.device.clone())
    }
}

// ============================================================================
// Component Cache
// ============================================================================

/// Cache for individual component tensors
/// 
/// Stores component embeddings keyed by content hash, enabling
/// incremental encoding where only changed files are re-processed.
pub struct ComponentCache {
    /// Cache directory
    cache_dir: PathBuf,
    
    /// Device for loaded tensors
    device: Device,
    
    /// In-memory index of cached components
    index: HashMap<String, CacheEntry>,
}

/// Entry in the component cache
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheEntry {
    /// Content hash (SHA256)
    pub content_hash: String,
    
    /// File path (for reference)
    pub file_path: String,
    
    /// Component IDs in this file
    pub component_ids: Vec<String>,
    
    /// Embedding mode used
    pub embedding_mode: String,
    
    /// When this was cached
    pub cached_at: DateTime<Utc>,
}

impl ComponentCache {
    /// Create a new component cache at the given directory
    pub fn new(cache_dir: impl AsRef<Path>, device: Device) -> Self {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        let index = Self::load_index(&cache_dir).unwrap_or_default();
        
        Self {
            cache_dir,
            device,
            index,
        }
    }
    
    /// Compute content hash for a file
    pub fn hash_file(path: &Path) -> Result<String> {
        let content = std::fs::read(path)
            .map_err(TensorCoreError::Io)?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    /// Compute content hash for a string
    pub fn hash_content(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
    
    /// Check if a file is cached (by content hash)
    pub fn is_cached(&self, content_hash: &str) -> bool {
        self.index.contains_key(content_hash)
    }
    
    /// Get cache entry for a content hash
    pub fn get_entry(&self, content_hash: &str) -> Option<&CacheEntry> {
        self.index.get(content_hash)
    }
    
    /// Save component tensors for a file
    pub fn save_components(
        &mut self,
        content_hash: &str,
        file_path: &Path,
        components: &[ComponentTensor],
        embedding_mode: &str,
    ) -> Result<()> {
        std::fs::create_dir_all(&self.cache_dir)
            .map_err(TensorCoreError::Io)?;
        
        // Save each component's hologram
        let hash_dir = self.cache_dir.join(&content_hash[..8]);
        std::fs::create_dir_all(&hash_dir)
            .map_err(TensorCoreError::Io)?;
        
        let mut tensors_data: HashMap<String, Vec<f32>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let component_ids: Vec<String> = components.iter()
            .map(|c| c.id.full_path())
            .collect();
        
        for (i, comp) in components.iter().enumerate() {
            let key = format!("comp_{}", i);
            let data: Vec<f32> = comp.hologram.flatten_all()?.to_vec1()?;
            shapes.insert(key.clone(), comp.hologram.dims().to_vec());
            tensors_data.insert(key, data);
        }
        
        // Save tensors
        if !tensors_data.is_empty() {
            let mut views: HashMap<String, TensorView> = HashMap::new();
            
            for (key, data) in &tensors_data {
                let shape = shapes.get(key).unwrap();
                let view = TensorView::new(
                    safetensors::tensor::Dtype::F32,
                    shape.clone(),
                    bytemuck::cast_slice(data),
                ).map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
                views.insert(key.clone(), view);
            }
            
            let tensor_path = hash_dir.join("tensors.safetensors");
            safetensors::serialize_to_file(views, &None, &tensor_path)
                .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        }
        
        // Update index
        let entry = CacheEntry {
            content_hash: content_hash.to_string(),
            file_path: file_path.to_string_lossy().to_string(),
            component_ids,
            embedding_mode: embedding_mode.to_string(),
            cached_at: Utc::now(),
        };
        
        self.index.insert(content_hash.to_string(), entry);
        self.save_index()?;
        
        Ok(())
    }
    
    /// Load cached component holograms for a file
    pub fn load_components(&self, content_hash: &str) -> Result<Vec<Tensor>> {
        let hash_dir = self.cache_dir.join(&content_hash[..8.min(content_hash.len())]);
        let tensor_path = hash_dir.join("tensors.safetensors");
        
        if !tensor_path.exists() {
            return Err(TensorCoreError::Config(
                format!("Cache miss: {}", content_hash)
            ));
        }
        
        let data = std::fs::read(&tensor_path)
            .map_err(TensorCoreError::Io)?;
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        
        let mut result = Vec::new();
        let mut i = 0;
        
        loop {
            let key = format!("comp_{}", i);
            match tensors.tensor(&key) {
                Ok(view) => {
                    let shape: Vec<usize> = view.shape().to_vec();
                    let float_data: Vec<f32> = bytemuck::cast_slice(view.data()).to_vec();
                    let tensor = Tensor::from_vec(float_data, shape.as_slice(), &self.device)
                        .map_err(|e| TensorCoreError::Tensor(e.to_string()))?;
                    result.push(tensor);
                    i += 1;
                }
                Err(_) => break,
            }
        }
        
        Ok(result)
    }
    
    /// Get cache stats
    pub fn stats(&self) -> CacheStats {
        let total_entries = self.index.len();
        let total_components: usize = self.index.values()
            .map(|e| e.component_ids.len())
            .sum();
        
        CacheStats {
            total_entries,
            total_components,
            cache_dir: self.cache_dir.clone(),
        }
    }
    
    /// Clear the cache
    pub fn clear(&mut self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)
                .map_err(TensorCoreError::Io)?;
        }
        self.index.clear();
        Ok(())
    }
    
    /// Load index from disk
    fn load_index(cache_dir: &Path) -> Result<HashMap<String, CacheEntry>> {
        let index_path = cache_dir.join("index.json");
        if !index_path.exists() {
            return Ok(HashMap::new());
        }
        
        let json = std::fs::read_to_string(&index_path)
            .map_err(TensorCoreError::Io)?;
        serde_json::from_str(&json)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))
    }
    
    /// Save index to disk
    fn save_index(&self) -> Result<()> {
        std::fs::create_dir_all(&self.cache_dir)
            .map_err(TensorCoreError::Io)?;
        
        let index_path = self.cache_dir.join("index.json");
        let json = serde_json::to_string_pretty(&self.index)
            .map_err(|e| TensorCoreError::Serialization(e.to_string()))?;
        std::fs::write(&index_path, json)
            .map_err(TensorCoreError::Io)
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached file entries
    pub total_entries: usize,
    /// Total number of cached component tensors
    pub total_components: usize,
    /// Path to the cache directory
    pub cache_dir: PathBuf,
}

// ============================================================================
// Incremental Encoding Support
// ============================================================================

/// Tracks which files need re-encoding
///
/// # Requirements
///
/// This type requires `git` to be available on the system PATH.
/// If git is not available, [`IncrementalState::analyze()`] will return an error.
/// In environments without git, treat all files as changed (full re-encoding).
pub struct IncrementalState {
    /// Previous commit hash
    pub previous_commit: Option<String>,
    
    /// Files that changed since previous commit
    pub changed_files: HashSet<PathBuf>,
    
    /// Files that can use cached tensors
    pub cached_files: HashSet<PathBuf>,
    
    /// Current commit hash
    pub current_commit: String,
}

impl IncrementalState {
    /// Analyze a project for incremental encoding
    pub fn analyze(root: &Path, cache: &ComponentCache) -> Result<Self> {
        let current_commit = get_current_commit(root)?;
        let previous_commit = Self::load_previous_commit(root)?;
        
        // Get all changed files
        let changed_files = if let Some(ref prev) = previous_commit {
            get_changed_files(root, prev)?
        } else {
            // First run - all files are "changed"
            get_all_rust_files(root)?
        };
        
        // Determine which files can use cache
        let mut cached_files = HashSet::new();
        let mut needs_encoding = HashSet::new();
        
        for file in &changed_files {
            if let Ok(hash) = ComponentCache::hash_file(file) {
                if cache.is_cached(&hash) {
                    cached_files.insert(file.clone());
                } else {
                    needs_encoding.insert(file.clone());
                }
            }
        }
        
        Ok(Self {
            previous_commit,
            changed_files: needs_encoding,
            cached_files,
            current_commit,
        })
    }
    
    /// Save current commit for next incremental run
    pub fn save_commit(&self, root: &Path) -> Result<()> {
        let state_dir = root.join(".an-ecosystem");
        std::fs::create_dir_all(&state_dir)
            .map_err(TensorCoreError::Io)?;
        
        let commit_file = state_dir.join("last_encoded_commit");
        std::fs::write(&commit_file, &self.current_commit)
            .map_err(TensorCoreError::Io)
    }
    
    /// Load previous commit from state file
    fn load_previous_commit(root: &Path) -> Result<Option<String>> {
        let commit_file = root.join(".an-ecosystem/last_encoded_commit");
        if commit_file.exists() {
            let commit = std::fs::read_to_string(&commit_file)
                .map_err(TensorCoreError::Io)?;
            Ok(Some(commit.trim().to_string()))
        } else {
            Ok(None)
        }
    }
    
    /// Get summary stats
    pub fn summary(&self) -> String {
        format!(
            "Incremental: {} changed, {} cached, {} total",
            self.changed_files.len(),
            self.cached_files.len(),
            self.changed_files.len() + self.cached_files.len()
        )
    }
}

/// Get current git commit
///
/// Requires `git` on PATH. Returns an error if git is unavailable or the
/// directory is not a git repository.
fn get_current_commit(root: &Path) -> Result<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .map_err(|e| TensorCoreError::Config(format!(
            "Git not available (required for incremental encoding): {}. \
             Install git or use full re-encoding instead.", e
        )))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(TensorCoreError::Config(format!(
            "Not a git repository at '{}': {}", root.display(), stderr.trim()
        )))
    }
}

/// Get files changed between commits
///
/// Returns only `.rs` files that differ between `since_commit` and HEAD.
fn get_changed_files(root: &Path, since_commit: &str) -> Result<HashSet<PathBuf>> {
    let output = std::process::Command::new("git")
        .args(["diff", "--name-only", since_commit, "HEAD"])
        .current_dir(root)
        .output()
        .map_err(|e| TensorCoreError::Config(format!(
            "Git diff failed (required for incremental encoding): {}", e
        )))?;

    let mut files = HashSet::new();

    if output.status.success() {
        let changed = String::from_utf8_lossy(&output.stdout);
        for line in changed.lines() {
            if line.ends_with(".rs") {
                files.insert(root.join(line));
            }
        }
    }

    Ok(files)
}

/// Get all Rust files in a project via `git ls-files`
///
/// Falls back to walkdir-based discovery if git is unavailable.
fn get_all_rust_files(root: &Path) -> Result<HashSet<PathBuf>> {
    let output = std::process::Command::new("git")
        .args(["ls-files", "*.rs"])
        .current_dir(root)
        .output();

    match output {
        Ok(output) if output.status.success() => {
            let mut files = HashSet::new();
            let all = String::from_utf8_lossy(&output.stdout);
            for line in all.lines() {
                files.insert(root.join(line));
            }
            Ok(files)
        }
        _ => {
            // Fallback: walk directory tree for .rs files
            let mut files = HashSet::new();
            for entry in walkdir::WalkDir::new(root)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "rs")
                    && !path.to_string_lossy().contains("/target/")
                {
                    files.insert(path.to_path_buf());
                }
            }
            Ok(files)
        }
    }
}

// ============================================================================
// Metadata Types
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OrganismMetadata {
    project_id: String,
    name: String,
    description: String,
    commit: String,
    updated_at: DateTime<Utc>,
    embedding_dim: usize,
    context_names: Vec<String>,
    federation: Vec<FederationBinding>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ContextMetadata {
    name: String,
    description: String,
    role: String,
    patterns: Vec<String>,
    api: Vec<String>,
    module_count: usize,
    component_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ComponentIndexEntry {
    id: String,
    kind: String,
    file: String,
    line_start: usize,
    line_end: usize,
    is_public: bool,
    #[serde(default)]
    module_idx: usize,
    #[serde(default)]
    component_idx: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_save_load_tensor() {
        let temp = TempDir::new().unwrap();
        let store = TensorStore::new(temp.path(), Device::Cpu);
        
        let tensor = Tensor::randn(0.0f32, 1.0, (1024,), &Device::Cpu).unwrap();
        let path = temp.path().join("test.safetensors");
        
        store.save_tensor(&tensor, &path).unwrap();
        let loaded = store.load_tensor(&path).unwrap();
        
        assert_eq!(tensor.dims(), loaded.dims());
    }
}
