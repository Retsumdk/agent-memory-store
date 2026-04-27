# agent-memory-store

Persistent vector-backed memory store for AI agents with semantic retrieval, context window management, and hierarchical memory tiers.

## Features

- **Semantic Retrieval** — Embedding-based similarity search across all memories
- **Hierarchical Memory Tiers** — Working, short-term, and long-term memory with configurable limits
- **Context Window Management** — Token-budget-aware retrieval that respects LLM context limits
- **TTL & Expiration** — Automatic memory expiration with configurable time-to-live
- **Memory Summarization** — Compress low-importance memories to save context space
- **Multi-Agent Support** — Isolated memory namespaces per agent ID
- **Persistence** — JSON-based storage with automatic save/restore
- **Tag-Based Filtering** — Organize memories with custom tags
- **Importance Scoring** — 0-10 importance system controls tier assignment and eviction order

## Installation

```bash
git clone https://github.com/Retsumdk/agent-memory-store.git
cd agent-memory-store
bun install
```

## Configuration

Create a `config.json` file:

```json
{
  "dataDir": "./memory-data",
  "maxContextTokens": 8000,
  "workingMemoryLimit": 50,
  "shortTermMemoryLimit": 200,
  "defaultTtlMs": 604800000,
  "similarityThreshold": 0.1,
  "embeddingDim": 384,
  "persistOnWrite": true
}
```

## Usage

### Store a Memory

```bash
bun run src/index.ts store \
  --agent "agent-123" \
  --content "The user prefers concise responses under 3 sentences" \
  --importance 8 \
  --tier long_term \
  --tags preference,user-style
```

### Retrieve Similar Memories

```bash
bun run src/index.ts retrieve \
  --agent "agent-123" \
  --query "What does the user like in responses?" \
  --limit 5 \
  --min-similarity 0.2
```

### Get Context Window

```bash
bun run src/index.ts context \
  --agent "agent-123" \
  --max-tokens 4000
```

### Memory Statistics

```bash
bun run src/index.ts stats --agent "agent-123"
```

### Evict Expired Memories

```bash
bun run src/index.ts evict --agent "agent-123"
```

### Summarize a Memory

```bash
bun run src/index.ts summarize \
  --agent "agent-123" \
  --memory "<memory-id>" \
  --target-tokens 100
```

## API Reference

### `AgentMemoryStore`

```typescript
const store = new AgentMemoryStore(config);
```

#### Methods

| Method | Description |
|--------|-------------|
| `store(agentId, content, options)` | Store a new memory |
| `retrieve(agentId, query, options)` | Retrieve by semantic similarity |
| `get(agentId, memoryId)` | Get a specific memory |
| `update(agentId, memoryId, updates)` | Update a memory |
| `delete(agentId, memoryId)` | Delete a memory |
| `getContext(agentId, maxTokens?)` | Get memories within token budget |
| `stats(agentId?)` | Get memory statistics |
| `summarize(agentId, memoryId, targetTokens?)` | Summarize and compress a memory |
| `evict(agentId)` | Remove expired memories |
| `export(filepath)` | Export all memories to JSON |
| `clear(agentId)` | Clear all memories for an agent |

### Store Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `importance` | `0-10` | `5` | Higher importance = lower eviction priority |
| `tier` | `working\|short_term\|long_term` | auto | Memory tier |
| `ttlMs` | `number\|null` | `defaultTtlMs` | Time-to-live in ms |
| `tags` | `string[]` | `[]` | Tags for filtering |

### Retrieval Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `limit` | `number` | `10` | Max results |
| `minSimilarity` | `0-1` | `0` | Minimum similarity score |
| `tier` | `string` | `"all"` | Filter by tier |
| `tags` | `string[]` | — | Filter by tags |
| `since` | `timestamp` | — | Only memories after timestamp |

## Architecture

```
AgentMemoryStore
├── EmbeddingService     — Generates/similarity-scores embeddings
├── PersistenceLayer     — JSON file read/write per agent
└── ContextWindowManager — Token budgeting for context fitting

Memory Tiers:
  working    → High-frequency, low-importance (limit: 50)
  short_term → Medium-priority memories (limit: 200)
  long_term  → High-importance, persistent (unlimited)
```

## License

MIT License

---

Built by [Retsumdk](https://github.com/Retsumdk)
