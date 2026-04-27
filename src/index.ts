#!/usr/bin/env bun
/**
 * agent-memory-store — Persistent vector-backed memory store for AI agents
 * 
 * Features:
 * - Embedding-based semantic memory storage
 * - Context window management with token budget
 * - Memory expiration with TTL support
 * - Hierarchical memory tiers (working, short-term, long-term)
 * - Similarity-based retrieval with configurable thresholds
 * - Memory summarization for compression
 * - Import/export for persistence
 * 
 * Built by Retsumdk
 */

import { Command } from "commander";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { v4 as uuidv4 } from "uuid";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface MemoryEntry {
  id: string;
  content: string;
  embedding: number[];
  createdAt: number;
  lastAccessedAt: number;
  accessCount: number;
  ttlMs: number | null;
  tier: "working" | "short_term" | "long_term";
  metadata: Record<string, unknown>;
  importance: number; // 0-10
  tags: string[];
  agentId: string;
}

export interface RetrievalResult {
  entry: MemoryEntry;
  similarity: number;
  rank: number;
}

export interface MemoryStats {
  total: number;
  byTier: Record<string, number>;
  byAgent: Record<string, number>;
  avgImportance: number;
  oldestEntry: number;
  newestEntry: number;
  totalTokens: number;
}

export interface StoreConfig {
  dataDir: string;
  maxContextTokens: number;
  workingMemoryLimit: number;
  shortTermMemoryLimit: number;
  defaultTtlMs: number | null;
  similarityThreshold: number;
  embeddingDim: number;
  persistOnWrite: boolean;
  autoSummarize: boolean;
  summarizeThreshold: number; // importance below this triggers potential summarization
}

export interface QueryOptions {
  limit?: number;
  minSimilarity?: number;
  agentId?: string;
  tags?: string[];
  tier?: "working" | "short_term" | "long_term" | "all";
  since?: number;
}

// ---------------------------------------------------------------------------
// Simple embedding (cosine similarity via TF-IDF-style hash vectors)
// For production, replace with OpenAI/Cohere embeddings
// ---------------------------------------------------------------------------

class EmbeddingService {
  private dim: number;
  private cache = new Map<string, number[]>();

  constructor(dim = 384) {
    this.dim = dim;
  }

  /** Generate a deterministic embedding from text content */
  async embed(text: string): Promise<number[]> {
    if (this.cache.has(text)) {
      return this.cache.get(text)!;
    }
    const vec = new Array(this.dim).fill(0);
    let seed = this.hashString(text);
    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i);
      const bucket = Math.abs((seed * (charCode + i * 31)) % this.dim);
      vec[bucket] += (charCode / 255) * Math.sin(seed + i);
      seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    }
    // L2 normalize
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    const normalized = norm > 0 ? vec.map(v => v / norm) : vec;
    this.cache.set(text, normalized);
    return normalized;
  }

  /** Cosine similarity between two vectors */
  cosineSim(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;
    let dot = 0, magA = 0, magB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      magA += a[i] * a[i];
      magB += b[i] * b[i];
    }
    const denom = Math.sqrt(magA) * Math.sqrt(magB);
    return denom > 0 ? dot / denom : 0;
  }

  private hashString(s: string): number {
    let h = 5381;
    for (let i = 0; i < s.length; i++) {
      h = ((h << 5) + h) + s.charCodeAt(i);
      h = h & 0xffffffff;
    }
    return Math.abs(h);
  }

  clearCache(): void {
    this.cache.clear();
  }
}

// ---------------------------------------------------------------------------
// Simple tokenizer for token counting
// ---------------------------------------------------------------------------

function estimateTokens(text: string): number {
  // Rough token estimation: ~4 chars per token for English
  return Math.ceil(text.length / 4);
}

// ---------------------------------------------------------------------------
// Memory Summarizer (simple extractive summarization)
// ---------------------------------------------------------------------------

function summarizeText(text: string, targetTokens = 100): string {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  if (sentences.length <= 2) return text;
  
  // Score sentences by length and character diversity
  const scored = sentences.map(s => ({
    text: s.trim(),
    score: s.trim().split(/\s+/).length * (new Set(s).size / s.length)
  }));
  scored.sort((a, b) => b.score - a.score);
  
  const selected: string[] = [];
  let tokens = 0;
  for (const s of scored) {
    const t = estimateTokens(s.text);
    if (tokens + t <= targetTokens) {
      selected.push(s.text);
      tokens += t;
    }
    if (tokens >= targetTokens) break;
  }
  
  return selected.join(" ").trim() || text.slice(0, targetTokens * 4);
}

// ---------------------------------------------------------------------------
// Persistence Layer
// ---------------------------------------------------------------------------

class PersistenceLayer {
  private dataDir: string;

  constructor(dataDir: string) {
    this.dataDir = dataDir;
    if (!existsSync(dataDir)) {
      mkdirSync(dataDir, { recursive: true });
    }
  }

  private filePath(agentId: string): string {
    return join(this.dataDir, `${this.sanitizeFilename(agentId)}.json`);
  }

  async load(agentId: string): Promise<MemoryEntry[]> {
    const fp = this.filePath(agentId);
    if (!existsSync(fp)) return [];
    try {
      const raw = readFileSync(fp, "utf-8");
      return JSON.parse(raw) as MemoryEntry[];
    } catch (e) {
      console.error(`Failed to load memory for ${agentId}: ${e}`);
      return [];
    }
  }

  async save(agentId: string, entries: MemoryEntry[]): Promise<void> {
    const fp = this.filePath(agentId);
    try {
      writeFileSync(fp, JSON.stringify(entries, null, 2), "utf-8");
    } catch (e) {
      console.error(`Failed to save memory for ${agentId}: ${e}`);
      throw e;
    }
  }

  async exportAll(dataDir: string): Promise<Record<string, MemoryEntry[]>> {
    const result: Record<string, MemoryEntry[]> = {};
    const { readdirSync } = await import("fs");
    const { readFileSync } = await import("fs");
    try {
      const files = readdirSync(dataDir).filter(f => f.endsWith(".json"));
      for (const file of files) {
        const agentId = file.replace(".json", "");
        const raw = readFileSync(join(dataDir, file), "utf-8");
        result[agentId] = JSON.parse(raw);
      }
    } catch (e) {
      console.error(`Export failed: ${e}`);
    }
    return result;
  }

  private sanitizeFilename(name: string): string {
    return name.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
  }
}

// ---------------------------------------------------------------------------
// Context Window Manager
// ---------------------------------------------------------------------------

class ContextWindowManager {
  constructor(private maxTokens: number) {}

  /** Estimate total tokens in a set of entries */
  estimateTokens(entries: MemoryEntry[]): number {
    return entries.reduce((sum, e) => sum + estimateTokens(e.content), 0);
  }

  /** Trim entries to fit within token budget, preserving importance order */
  fitToContext(entries: MemoryEntry[], maxTokens?: number): MemoryEntry[] {
    const limit = maxTokens ?? this.maxTokens;
    const sorted = [...entries].sort((a, b) => b.importance - a.importance);
    const result: MemoryEntry[] = [];
    let tokens = 0;
    for (const entry of sorted) {
      const entryTokens = estimateTokens(entry.content);
      if (tokens + entryTokens <= limit) {
        result.push(entry);
        tokens += entryTokens;
      }
      if (tokens >= limit) break;
    }
    return result;
  }

  /** Check if adding new content would exceed budget */
  wouldExceed(entries: MemoryEntry[], newContentTokens: number): boolean {
    return this.estimateTokens(entries) + newContentTokens > this.maxTokens;
  }
}

// ---------------------------------------------------------------------------
// Main Memory Store
// ---------------------------------------------------------------------------

export class AgentMemoryStore {
  private memories = new Map<string, MemoryEntry[]>();
  private embeddings = new EmbeddingService();
  private persistence: PersistenceLayer;
  private contextManager: ContextWindowManager;

  constructor(private config: StoreConfig) {
    this.persistence = new PersistenceLayer(config.dataDir);
    this.contextManager = new ContextWindowManager(config.maxContextTokens);
  }

  // -------------------------------------------------------------------------
  // Store Operations
  // -------------------------------------------------------------------------

  /**
   * Store a new memory entry for an agent.
   * Automatically assigns to the appropriate tier based on importance.
   */
  async store(
    agentId: string,
    content: string,
    options: {
      importance?: number;
      ttlMs?: number | null;
      tier?: "working" | "short_term" | "long_term";
      tags?: string[];
      metadata?: Record<string, unknown>;
    } = {}
  ): Promise<MemoryEntry> {
    const {
      importance = 5,
      ttlMs = this.config.defaultTtlMs,
      tier,
      tags = [],
      metadata = {}
    } = options;

    // Determine tier if not specified
    let assignedTier: "working" | "short_term" | "long_term" =
      tier ?? (importance >= 8 ? "long_term" : importance >= 4 ? "short_term" : "working");

    // Enforce tier limits
    const entries = await this.getEntries(agentId);
    await this.enforceTierLimits(agentId, entries, assignedTier);

    const embedding = await this.embeddings.embed(content);
    const now = Date.now();

    const entry: MemoryEntry = {
      id: uuidv4(),
      content,
      embedding,
      createdAt: now,
      lastAccessedAt: now,
      accessCount: 0,
      ttlMs,
      tier: assignedTier,
      metadata,
      importance,
      tags,
      agentId,
    };

    const allEntries = await this.getEntries(agentId);
    allEntries.push(entry);
    await this.save(agentId, allEntries);

    console.log(
      `[agent-memory-store] Stored ${entry.id.slice(0, 8)} for ${agentId} ` +
      `(tier=${assignedTier}, importance=${importance}, tokens≈${estimateTokens(content)})`
    );

    return entry;
  }

  /**
   * Retrieve memories by semantic similarity to a query.
   */
  async retrieve(
    agentId: string,
    query: string,
    options: QueryOptions = {}
  ): Promise<RetrievalResult[]> {
    const {
      limit = 10,
      minSimilarity = 0,
      agentId: _filterAgentId,
      tags,
      tier = "all",
      since,
    } = options;

    let entries = await this.getEntries(agentId);
    const queryEmbedding = await this.embeddings.embed(query);
    const now = Date.now();

    // Filter entries
    entries = entries.filter(e => {
      // TTL check
      if (e.ttlMs !== null && e.createdAt + e.ttlMs < now) return false;
      // Time filter
      if (since && e.createdAt < since) return false;
      // Tag filter
      if (tags && tags.length > 0 && !tags.some(t => e.tags.includes(t))) return false;
      // Tier filter
      if (tier !== "all" && e.tier !== tier) return false;
      return true;
    });

    // Score by similarity
    const scored = entries.map(e => ({
      entry: e,
      similarity: this.embeddings.cosineSim(queryEmbedding, e.embedding),
    }));

    // Sort by similarity (primary) then importance (secondary)
    scored.sort((a, b) => {
      if (Math.abs(a.similarity - b.similarity) > 0.01) {
        return b.similarity - a.similarity;
      }
      return b.entry.importance - a.entry.importance;
    });

    // Filter by minimum similarity
    const filtered = scored.filter(s => s.similarity >= minSimilarity);

    // Update access stats
    const results = filtered.slice(0, limit);
    for (const r of results) {
      r.entry.lastAccessedAt = now;
      r.entry.accessCount++;
    }
    await this.save(agentId, entries);

    return results.map((r, i) => ({ ...r, rank: i + 1 }));
  }

  /**
   * Get a specific memory entry by ID.
   */
  async get(agentId: string, memoryId: string): Promise<MemoryEntry | null> {
    const entries = await this.getEntries(agentId);
    const entry = entries.find(e => e.id === memoryId);
    if (entry) {
      entry.lastAccessedAt = Date.now();
      entry.accessCount++;
      await this.save(agentId, entries);
    }
    return entry ?? null;
  }

  /**
   * Delete a memory entry.
   */
  async delete(agentId: string, memoryId: string): Promise<boolean> {
    const entries = await this.getEntries(agentId);
    const idx = entries.findIndex(e => e.id === memoryId);
    if (idx === -1) return false;
    entries.splice(idx, 1);
    await this.save(agentId, entries);
    console.log(`[agent-memory-store] Deleted ${memoryId} for ${agentId}`);
    return true;
  }

  /**
   * Update an existing memory entry.
   */
  async update(
    agentId: string,
    memoryId: string,
    updates: Partial<Pick<MemoryEntry, "content" | "importance" | "tags" | "metadata" | "tier" | "ttlMs">>
  ): Promise<MemoryEntry | null> {
    const entries = await this.getEntries(agentId);
    const entry = entries.find(e => e.id === memoryId);
    if (!entry) return null;

    if (updates.content !== undefined) {
      entry.content = updates.content;
      entry.embedding = await this.embeddings.embed(updates.content);
    }
    if (updates.importance !== undefined) entry.importance = updates.importance;
    if (updates.tags !== undefined) entry.tags = updates.tags;
    if (updates.metadata !== undefined) entry.metadata = { ...entry.metadata, ...updates.metadata };
    if (updates.tier !== undefined) entry.tier = updates.tier;
    if (updates.ttlMs !== undefined) entry.ttlMs = updates.ttlMs;

    await this.save(agentId, entries);
    return entry;
  }

  /**
   * Get all memories within a context window, fitted to token budget.
   */
  async getContext(agentId: string, maxTokens?: number): Promise<MemoryEntry[]> {
    const entries = await this.getEntries(agentId);
    const now = Date.now();
    const valid = entries.filter(e => e.ttlMs === null || e.createdAt + e.ttlMs >= now);
    return this.contextManager.fitToContext(valid, maxTokens);
  }

  /**
   * Get statistics about stored memories.
   */
  async stats(agentId?: string): Promise<MemoryStats | Record<string, MemoryStats>> {
    if (agentId) {
      return this.computeStats(await this.getEntries(agentId));
    }
    // Aggregate stats across all agents
    const allAgents = [...this.memories.keys()];
    const agentIds = [...new Set(allAgents)];
    const result: Record<string, MemoryStats> = {};
    for (const aid of agentIds) {
      result[aid] = this.computeStats(await this.getEntries(aid));
    }
    return result;
  }

  /**
   * Summarize a memory entry and replace its content.
   */
  async summarize(agentId: string, memoryId: string, targetTokens = 100): Promise<string | null> {
    const entries = await this.getEntries(agentId);
    const entry = entries.find(e => e.id === memoryId);
    if (!entry) return null;

    const summary = summarizeText(entry.content, targetTokens);
    entry.content = `[Summary] ${summary}`;
    entry.embedding = await this.embeddings.embed(entry.content);
    entry.importance = Math.max(1, entry.importance - 1); // reduce importance after summarization

    await this.save(agentId, entries);
    console.log(`[agent-memory-store] Summarized ${memoryId} (${entry.importance} importance)`);
    return summary;
  }

  /**
   * Evict expired memories and enforce tier limits.
   */
  async evict(agentId: string): Promise<number> {
    const entries = await this.getEntries(agentId);
    const now = Date.now();
    const before = entries.length;
    
    const remaining = entries.filter(e => {
      if (e.ttlMs !== null && e.createdAt + e.ttlMs < now) {
        console.log(`[agent-memory-store] Evicting expired ${e.id}`);
        return false;
      }
      return true;
    });

    await this.save(agentId, remaining);
    return before - remaining.length;
  }

  /**
   * Export all memories to a JSON file.
   */
  async export(filepath: string): Promise<void> {
    const all = await this.persistence.exportAll(this.config.dataDir);
    writeFileSync(filepath, JSON.stringify(all, null, 2), "utf-8");
    console.log(`[agent-memory-store] Exported to ${filepath}`);
  }

  /**
   * Clear all memories for an agent.
   */
  async clear(agentId: string): Promise<void> {
    await this.save(agentId, []);
    console.log(`[agent-memory-store] Cleared all memories for ${agentId}`);
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  private async getEntries(agentId: string): Promise<MemoryEntry[]> {
    if (!this.memories.has(agentId)) {
      const loaded = await this.persistence.load(agentId);
      this.memories.set(agentId, loaded);
    }
    return this.memories.get(agentId)!;
  }

  private async save(agentId: string, entries: MemoryEntry[]): Promise<void> {
    this.memories.set(agentId, entries);
    if (this.config.persistOnWrite) {
      await this.persistence.save(agentId, entries);
    }
  }

  private async enforceTierLimits(
    agentId: string,
    entries: MemoryEntry[],
    newTier: "working" | "short_term" | "long_term"
  ): Promise<void> {
    const limits: Record<string, number> = {
      working: this.config.workingMemoryLimit,
      short_term: this.config.shortTermMemoryLimit,
      long_term: Infinity,
    };
    const limit = limits[newTier] ?? Infinity;
    const tierEntries = entries.filter(e => e.tier === newTier);
    if (tierEntries.length >= limit) {
      // Evict least important / oldest entry in this tier
      tierEntries.sort((a, b) => {
        if (a.importance !== b.importance) return a.importance - b.importance;
        return a.lastAccessedAt - b.lastAccessedAt;
      });
      const toEvict = tierEntries[0];
      const idx = entries.findIndex(e => e.id === toEvict.id);
      if (idx !== -1) {
        entries.splice(idx, 1);
        console.log(`[agent-memory-store] Tier limit reached, evicted ${toEvict.id} from ${newTier}`);
      }
    }
  }

  private computeStats(entries: MemoryEntry[]): MemoryStats {
    const now = Date.now();
    const valid = entries.filter(e => e.ttlMs === null || e.createdAt + e.ttlMs >= now);
    const byTier: Record<string, number> = {};
    const byAgent: Record<string, number> = {};
    let totalTokens = 0;
    let totalImportance = 0;
    let oldest = Infinity;
    let newest = 0;

    for (const e of valid) {
      byTier[e.tier] = (byTier[e.tier] ?? 0) + 1;
      byAgent[e.agentId] = (byAgent[e.agentId] ?? 0) + 1;
      totalTokens += estimateTokens(e.content);
      totalImportance += e.importance;
      if (e.createdAt < oldest) oldest = e.createdAt;
      if (e.createdAt > newest) newest = e.createdAt;
    }

    return {
      total: valid.length,
      byTier,
      byAgent,
      avgImportance: valid.length > 0 ? totalImportance / valid.length : 0,
      oldestEntry: valid.length > 0 ? oldest : 0,
      newestEntry: valid.length > 0 ? newest : 0,
      totalTokens,
    };
  }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG: StoreConfig = {
  dataDir: "./memory-data",
  maxContextTokens: 8000,
  workingMemoryLimit: 50,
  shortTermMemoryLimit: 200,
  defaultTtlMs: 7 * 24 * 60 * 60 * 1000, // 7 days
  similarityThreshold: 0.1,
  embeddingDim: 384,
  persistOnWrite: true,
  autoSummarize: false,
  summarizeThreshold: 3,
};

function loadConfig(cfgPath?: string): StoreConfig {
  if (cfgPath && existsSync(cfgPath)) {
    try {
      return { ...DEFAULT_CONFIG, ...JSON.parse(readFileSync(cfgPath, "utf-8")) };
    } catch {
      console.error(`Failed to parse config: ${cfgPath}`);
    }
  }
  return { ...DEFAULT_CONFIG };
}

async function runCLI() {
  const program = new Command();
  
  program
    .name("agent-memory-store")
    .description("Persistent vector-backed memory store for AI agents")
    .version("1.0.0")
    .option("-c, --config <path>", "Config file path", "config.json")
    .option("-v, --verbose", "Verbose output");

  program
    .command("store")
    .description("Store a new memory")
    .requiredOption("-a, --agent <id>", "Agent ID")
    .requiredOption("-C, --content <text>", "Memory content")
    .option("-i, --importance <0-10>", "Importance score (0-10)", "5")
    .option("--tier <working|short_term|long_term>", "Memory tier")
    .option("--tags <tags...>", "Tags")
    .option("--ttl <ms>", "TTL in milliseconds")
    .action(async (opts) => {
      const config = loadConfig(program.opts().config);
      const store = new AgentMemoryStore(config);
      const entry = await store.store(opts.agent, opts.content, {
        importance: parseInt(opts.importance),
        tier: opts.tier as any,
        tags: opts.tags,
        ttlMs: opts.ttl ? parseInt(opts.ttl) : config.defaultTtlMs,
      });
      console.log(JSON.stringify(entry, null, 2));
    });

  program
    .command("retrieve")
    .description("Retrieve memories by semantic similarity")
    .requiredOption("-a, --agent <id>", "Agent ID")
    .requiredOption("-q, --query <text>", "Query text")
    .option("-l, --limit <n>", "Max results", "10")
    .option("-s, --min-similarity <n>", "Minimum similarity", "0")
    .option("--tier <tier>", "Filter by tier")
    .option("--tags <tags...>", "Filter by tags")
    .action(async (opts) => {
      const config = loadConfig(program.opts().config);
      const store = new AgentMemoryStore(config);
      const results = await store.retrieve(opts.agent, opts.query, {
        limit: parseInt(opts.limit),
        minSimilarity: parseFloat(opts.minSimilarity),
        tier: opts.tier as any,
        tags: opts.tags,
      });
      console.log(JSON.stringify(results, null, 2));
    });

  program
    .command("stats")
    .description("Get memory statistics")
    .option("-a, --agent <id>", "Agent ID (optional for global)")
    .action(async (opts) => {
      const config = loadConfig(program.opts().config);
      const store = new AgentMemoryStore(config);
      const stats = await store.stats(opts.agent);
      console.log(JSON.stringify(stats, null, 2));
    });

  program
    .command("evict")
    .description("Evict expired memories")
    .requiredOption("-a, --agent <id>", "Agent ID")
    .action(async (opts) => {
      const config = loadConfig(program.opts().config);
      const store = new AgentMemoryStore(config);
      const count = await store.evict(opts.agent);
      console.log(`Evicted ${count} entries`);
    });

  program
    .command("context")
    .description("Get context window for agent")
    .requiredOption("-a, --agent <id>", "Agent ID")
    .option("-t, --max-tokens <n>", "Max tokens", "8000")
    .action(async (opts) => {
      const config = loadConfig(program.opts().config);
      const store = new AgentMemoryStore(config);
      const ctx = await store.getContext(opts.agent, parseInt(opts.maxTokens));
      console.log(JSON.stringify(ctx, null, 2));
    });

  program
    .command("delete")
    .description("Delete a memory entry")
    .requiredOption("-a, --agent <id>", "Agent ID")
    .requiredOption("-m, --memory <id>", "Memory ID")
    .action(async (opts) => {
      const config = loadConfig(program.opts().config);
      const store = new AgentMemoryStore(config);
      const deleted = await store.delete(opts.agent, opts.memory);
      console.log(deleted ? "Deleted" : "Not found");
    });

  program
    .command("summarize")
    .description("Summarize a memory entry")
    .requiredOption("-a, --agent <id>", "Agent ID")
    .requiredOption("-m, --memory <id>", "Memory ID")
    .option("-t, --target-tokens <n>", "Target token count", "100")
    .action(async (opts) => {
      const config = loadConfig(program.opts().config);
      const store = new AgentMemoryStore(config);
      const summary = await store.summarize(opts.agent, opts.memory, parseInt(opts.targetTokens));
      console.log(summary ?? "Not found");
    });

  await program.parseAsync(process.argv);
}

// Run as CLI or import as module
if (import.meta.main) {
  runCLI().catch(e => {
    console.error(e);
    process.exit(1);
  });
}
