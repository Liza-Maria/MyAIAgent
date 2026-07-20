# Stage 6: Vector Store

**Goal:** Build an in-memory store that holds documents with their embeddings and finds the most similar ones for a query. This connects your two Stage 5 pieces: `cosine_similarity` gives the score, `OllamaEmbedder` produces the vectors.

Work in `crates/agent-core/src/rag/store.rs`. Do the tasks in order — each one compiles and is testable on its own.

### Task 1 — The `Document` struct

Define a struct that holds one stored item: an `id: String`, the original `text: String`, and its `embedding: Vec<f32>`.

> **Hint:** You need the original text because that's what the agent will eventually paste into the LLM prompt — the embedding is only for searching, you can't turn it back into text.

### Task 2 — The `VectorStore` struct with `new` and `add`

Create a `VectorStore` that owns a `Vec<Document>`. Give it a `new()` and an `add(&mut self, document: Document)`.

> **Hint:** Why `&mut self` here, when `embed` in Stage 5 only needed `&self`? Answer that question for yourself before moving on.

### Task 3 — Reject wrong-sized embeddings in `add`

All embeddings in a store must have the same number of dimensions, otherwise similarity is meaningless. Change `add` to return `Result<(), StoreError>` and reject a document whose embedding length doesn't match the documents already stored (also reject empty embeddings). Define `StoreError` in `rag/mod.rs` next to `EmbedError`.

> **Hint:** You already know the pattern — look at how `EmbedError` uses `thiserror`. Your error message should include both the expected and the actual dimension, like your `Api { status, body }` variant carries data.
>
> **Hint:** The first document added defines the expected dimension. Where can you read it from later? `self.documents.first()` gives you an `Option<&Document>`.

### Task 4 — `search`

```rust
pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchResult>
```

Define a small `SearchResult` struct (document id, text, and `score: f32`). Score **every** stored document against the query with your `cosine_similarity`, sort best-first, and return at most `top_k` results.

> **Hint:** Sorting by an `f32` is the one tricky bit: `sort_by(|a, b| a.score.cmp(&b.score))` will not compile, because floats don't implement `Ord` (thanks to `NaN`). Look up `partial_cmp` and `Ordering` — and note `.reverse()` exists on `Ordering`.
>
> **Hint:** For "at most top_k": `truncate` is simpler than an if-statement.

### Task 5 — Tests

Write these four (plain `#[test]`, no async, no wiremock — the store never touches the network):

1. `search_returns_best_match_first` — add 3 documents with hand-made small vectors (2–3 dimensions are enough, like your cosine tests), search with a query, assert the ids come back in the right order.
2. `search_on_empty_store_returns_empty`
3. `search_respects_top_k` — 3 documents, `top_k = 2`, expect 2 results.
4. `add_rejects_mismatched_dimensions` — add a 3-dim document, then try a 2-dim one, assert the error. (`assert!(matches!(...))` — you used it in the embedder test.)

> **Hint for test 1:** Pick vectors where you *know* the answer geometrically. If the query is `[1.0, 0.0]`, then `[0.9, 0.1]` beats `[0.0, 1.0]`. Don't invent 10-dimensional numbers you can't reason about.

### Done when

- [ ] `cargo test -p agent-core store` — all green
- [ ] `cargo build` — no new warnings
- [ ] `search` never panics, for any input (empty store, huge `top_k`, wrong-sized query)

### Bonus (only if the rest is done)

Wrong-sized *query* in `search`: right now `cosine_similarity` silently returns `0.0` for it. Is silently returning no-good-matches the behavior you want, or should `search` also return a `Result`? There's no single right answer — pick one and write a comment defending your choice.
