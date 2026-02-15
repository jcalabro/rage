//! LRU cache for decompressed Syzygy table blocks.
//!
//! Caches recently accessed decompressed blocks to avoid repeated I/O
//! and decompression. Thread-safe via RwLock (probing is infrequent
//! relative to search node rate).

use std::collections::HashMap;
use std::sync::RwLock;

/// Key identifying a cached block: (table_name, subtable_index, block_index).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    table_name: String,
    subtable: usize,
    block: usize,
}

/// A cached decompressed block.
struct CacheEntry {
    data: Vec<u8>,
    /// LRU generation (higher = more recent).
    access_gen: u64,
}

/// LRU cache for decompressed table blocks.
pub struct BlockCache {
    inner: RwLock<CacheInner>,
}

struct CacheInner {
    entries: HashMap<CacheKey, CacheEntry>,
    next_gen: u64,
    max_entries: usize,
}

impl CacheInner {
    fn bump_gen(&mut self) -> u64 {
        self.next_gen += 1;
        self.next_gen
    }
}

impl BlockCache {
    /// Create a new cache with the given maximum number of entries.
    pub fn new(max_entries: usize) -> Self {
        Self {
            inner: RwLock::new(CacheInner {
                entries: HashMap::new(),
                next_gen: 0,
                max_entries,
            }),
        }
    }

    /// Try to retrieve a cached block.
    pub fn get(&self, table_name: &str, subtable: usize, block: usize) -> Option<Vec<u8>> {
        let key = CacheKey {
            table_name: table_name.to_string(),
            subtable,
            block,
        };

        if let Ok(mut inner) = self.inner.write() {
            let g = inner.bump_gen();
            if let Some(entry) = inner.entries.get_mut(&key) {
                entry.access_gen = g;
                return Some(entry.data.clone());
            }
        }
        None
    }

    /// Insert a decompressed block into the cache.
    pub fn insert(&self, table_name: &str, subtable: usize, block: usize, data: Vec<u8>) {
        let key = CacheKey {
            table_name: table_name.to_string(),
            subtable,
            block,
        };

        if let Ok(mut inner) = self.inner.write() {
            let g = inner.bump_gen();

            // Evict LRU entry if at capacity
            if inner.entries.len() >= inner.max_entries && !inner.entries.contains_key(&key) {
                evict_lru(&mut inner.entries);
            }

            inner.entries.insert(
                key,
                CacheEntry {
                    data,
                    access_gen: g,
                },
            );
        }
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.write() {
            inner.entries.clear();
            inner.next_gen = 0;
        }
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .map(|inner| inner.entries.len())
            .unwrap_or(0)
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Evict the least recently used entry from the map.
fn evict_lru(entries: &mut HashMap<CacheKey, CacheEntry>) {
    if let Some(lru_key) = entries
        .iter()
        .min_by_key(|(_, v)| v.access_gen)
        .map(|(k, _)| k.clone())
    {
        entries.remove(&lru_key);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn cache_insert_and_get() {
        let cache = BlockCache::new(10);
        cache.insert("KRvK", 0, 0, vec![1, 2, 3]);
        let result = cache.get("KRvK", 0, 0);
        assert_eq!(result, Some(vec![1, 2, 3]));
    }

    #[test]
    fn cache_miss() {
        let cache = BlockCache::new(10);
        assert_eq!(cache.get("KRvK", 0, 0), None);
    }

    #[test]
    fn cache_eviction() {
        let cache = BlockCache::new(2);
        cache.insert("A", 0, 0, vec![1]);
        cache.insert("B", 0, 0, vec![2]);
        assert_eq!(cache.len(), 2);

        // Insert third entry, should evict the LRU (A)
        cache.insert("C", 0, 0, vec![3]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get("A", 0, 0), None); // evicted
        assert!(cache.get("B", 0, 0).is_some()); // still present
        assert!(cache.get("C", 0, 0).is_some()); // just inserted
    }

    #[test]
    fn cache_lru_ordering() {
        let cache = BlockCache::new(2);
        cache.insert("A", 0, 0, vec![1]);
        cache.insert("B", 0, 0, vec![2]);

        // Access A to make it more recent
        cache.get("A", 0, 0);

        // Insert C - should evict B (least recently used)
        cache.insert("C", 0, 0, vec![3]);
        assert!(cache.get("A", 0, 0).is_some()); // recently accessed
        assert_eq!(cache.get("B", 0, 0), None); // evicted
        assert!(cache.get("C", 0, 0).is_some());
    }

    #[test]
    fn cache_clear() {
        let cache = BlockCache::new(10);
        cache.insert("A", 0, 0, vec![1]);
        cache.insert("B", 0, 0, vec![2]);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_update_existing() {
        let cache = BlockCache::new(10);
        cache.insert("A", 0, 0, vec![1]);
        cache.insert("A", 0, 0, vec![99]);
        assert_eq!(cache.get("A", 0, 0), Some(vec![99]));
        assert_eq!(cache.len(), 1);
    }
}
