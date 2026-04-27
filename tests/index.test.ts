import { describe, test, expect, beforeEach } from "bun:test";
import { AgentMemoryStore, type StoreConfig } from "../src/index";

const testConfig: StoreConfig = {
  dataDir: "/tmp/test-memory-data",
  maxContextTokens: 2000,
  workingMemoryLimit: 10,
  shortTermMemoryLimit: 20,
  defaultTtlMs: null,
  similarityThreshold: 0.1,
  embeddingDim: 128,
  persistOnWrite: false,
  autoSummarize: false,
  summarizeThreshold: 3,
};

describe("AgentMemoryStore", () => {
  let store: AgentMemoryStore;

  beforeEach(async () => {
    store = new AgentMemoryStore(testConfig);
    await store.clear("test-agent");
  });

  test("stores and retrieves a memory", async () => {
    const entry = await store.store("test-agent", "The sky is blue", {
      importance: 7,
    });
    expect(entry.id).toBeDefined();
    expect(entry.content).toBe("The sky is blue");
    expect(entry.importance).toBe(7);
    expect(entry.tier).toBe("short_term");
  });

  test("retrieves by similarity", async () => {
    await store.store("test-agent", "The sky is blue and clear");
    await store.store("test-agent", "The ocean is deep and blue");
    await store.store("test-agent", "Dogs are loyal companions");

    const results = await store.retrieve("test-agent", "blue color", { limit: 2 });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].similarity).toBeGreaterThan(0);
  });

  test("updates a memory", async () => {
    const entry = await store.store("test-agent", "Original content");
    const updated = await store.update("test-agent", entry.id, {
      content: "Updated content",
      importance: 9,
    });
    expect(updated?.content).toBe("Updated content");
    expect(updated?.importance).toBe(9);
  });

  test("deletes a memory", async () => {
    const entry = await store.store("test-agent", "To be deleted");
    const deleted = await store.delete("test-agent", entry.id);
    expect(deleted).toBe(true);

    const found = await store.get("test-agent", entry.id);
    expect(found).toBeNull();
  });

  test("respects tier limits", async () => {
    const cfg = { ...testConfig, workingMemoryLimit: 2 };
    const limitedStore = new AgentMemoryStore(cfg);

    await limitedStore.store("agent-x", "Memory 1", { tier: "working", importance: 5 });
    await limitedStore.store("agent-x", "Memory 2", { tier: "working", importance: 5 });
    // Third working memory should evict one
    await limitedStore.store("agent-x", "Memory 3", { tier: "working", importance: 5 });

    const ctx = await limitedStore.getContext("agent-x");
    const workingMemories = ctx.filter(e => e.tier === "working");
    expect(workingMemories.length).toBeLessThanOrEqual(2);
  });

  test("computes correct stats", async () => {
    await store.store("agent-y", "Memory A", { importance: 5 });
    await store.store("agent-y", "Memory B", { importance: 8 });
    await store.store("agent-y", "Memory C", { importance: 3 });

    const stats = await store.stats("agent-y");
    expect(stats.total).toBe(3);
    expect(stats.avgImportance).toBeCloseTo(5.33, 1);
  });

  test("filters by tags on retrieval", async () => {
    await store.store("test-agent", "Has tag A", { tags: ["tag-a"] });
    await store.store("test-agent", "Has tag B", { tags: ["tag-b"] });
    await store.store("test-agent", "Has both tags", { tags: ["tag-a", "tag-b"] });

    const results = await store.retrieve("test-agent", "something", {
      tags: ["tag-a"],
    });
    expect(results.length).toBe(2);
  });

  test("context fits to token budget", async () => {
    const cfg = { ...testConfig, maxContextTokens: 50 };
    const budgetStore = new AgentMemoryStore(cfg);

    await budgetStore.store("agent-z", "This is a very long piece of content that should be trimmed when context budget is tight");
    await budgetStore.store("agent-z", "Short");

    const ctx = await budgetStore.getContext("agent-z");
    const totalTokens = ctx.reduce((sum, e) => sum + Math.ceil(e.content.length / 4), 0);
    expect(totalTokens).toBeLessThanOrEqual(60); // Some slack for rounding
  });
});
