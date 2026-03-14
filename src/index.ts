export interface MemoryEntry {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
  priority: number;
  createdAt?: number;
  lastAccessed?: number;
}

export interface MemoryConfig {
  maxSize?: number;
  decayEnabled?: boolean;
  decayThreshold?: number;
  retrievalLimit?: number;
  embeddingDimension?: number;
}

export interface SearchOptions {
  limit?: number;
  minRelevance?: number;
}

export interface SearchResult {
  entry: MemoryEntry;
  relevance: number;
}

export class AgentMemory {
  private memories: Map<string, MemoryEntry> = new Map();
  private config: Required<MemoryConfig>;
  private accessLog: Map<string, number> = new Map();

  constructor(config: MemoryConfig = {}) {
    this.config = {
      maxSize: config.maxSize ?? 1000,
      decayEnabled: config.decayEnabled ?? true,
      decayThreshold: config.decayThreshold ?? 3600000,
      retrievalLimit: config.retrievalLimit ?? 10,
      embeddingDimension: config.embeddingDimension ?? 384,
    };
  }

  async store(entry: MemoryEntry): Promise<void> {
    const now = Date.now();
    const memoryEntry: MemoryEntry = {
      ...entry,
      createdAt: entry.createdAt ?? now,
      lastAccessed: now,
    };

    if (this.memories.size >= this.config.maxSize && !this.memories.has(entry.id)) {
      await this.evict();
    }

    this.memories.set(entry.id, memoryEntry);
    this.accessLog.set(entry.id, now);
  }

  async retrieve(id: string): Promise<MemoryEntry | null> {
    const entry = this.memories.get(id);
    if (!entry) return null;

    const now = Date.now();
    entry.lastAccessed = now;
    this.accessLog.set(id, now);

    return entry;
  }

  async remove(id: string): Promise<boolean> {
    this.accessLog.delete(id);
    return this.memories.delete(id);
  }

  async update(id: string, updates: Partial<MemoryEntry>): Promise<MemoryEntry | null> {
    const entry = this.memories.get(id);
    if (!entry) return null;

    const updated = { ...entry, ...updates };
    this.memories.set(id, updated);
    return updated;
  }

  async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    const limit = options.limit ?? this.config.retrievalLimit;
    const results: SearchResult[] = [];

    const queryLower = query.toLowerCase();
    const queryTerms = queryLower.split(/\s+/);

    for (const [id, entry] of this.memories) {
      let relevance = 0;
      const contentLower = entry.content.toLowerCase();

      for (const term of queryTerms) {
        if (contentLower.includes(term)) {
          relevance += 1;
        }
      }

      if (entry.metadata?.tags && Array.isArray(entry.metadata.tags)) {
        for (const tag of entry.metadata.tags) {
          if (typeof tag === 'string' && queryTerms.some(t => tag.toLowerCase().includes(t))) {
            relevance += 0.5;
          }
        }
      }

      relevance += entry.priority * 0.1;

      const lastAccessed = this.accessLog.get(id) ?? entry.createdAt ?? 0;
      const age = Date.now() - lastAccessed;
      if (this.config.decayEnabled && age > this.config.decayThreshold) {
        const decayFactor = Math.max(0.1, 1 - (age - this.config.decayThreshold) / this.config.decayThreshold);
        relevance *= decayFactor;
      }

      if (relevance > (options.minRelevance ?? 0)) {
        results.push({ entry, relevance });
      }
    }

    results.sort((a, b) => b.relevance - a.relevance);
    return results.slice(0, limit);
  }

  async evict(): Promise<void> {
    if (this.memories.size === 0) return;

    let oldest: string | null = null;
    let oldestTime = Infinity;

    for (const [id, time] of this.accessLog) {
      if (time < oldestTime) {
        oldestTime = time;
        oldest = id;
      }
    }

    if (!oldest) {
      const entries = Array.from(this.memories.keys());
      oldest = entries[0];
    }

    if (oldest) {
      await this.remove(oldest);
    }
  }

  async clear(): Promise<void> {
    this.memories.clear();
    this.accessLog.clear();
  }

  size(): number {
    return this.memories.size;
  }

  getConfig(): Required<MemoryConfig> {
    return { ...this.config };
  }
}

export default AgentMemory;
