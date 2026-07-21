# Stage 7: Retriever

**Goal:** Build the piece that connects your embedder and your vector store. Until now they don't know about each other: the `OllamaEmbedder` turns text into a vector, the `VectorStore` searches vectors — but nobody embeds a *query* and then searches with it. That's the `Retriever`, and it's the "R" in RAG.

This is your first component that composes two async pieces, so take the tasks slowly.

Work in a new file `crates/agent-core/src/rag/retriever.rs`. Remember to add `pub mod retriever;` in `rag/mod.rs`.

---

## Warm-up — fix three things from Stage 6 first

You'll need these fixed before the Retriever can even compile, so do them now. Each is a one-liner.

### Warm-up A — make `SearchResult` fields public
In `rag/mod.rs`, the fields of `SearchResult` (`id`, `text`, `score`) have no `pub`. The Retriever lives in a different position in the module tree and will need to *read* `.score` and `.text`. Add `pub` to all three.

> **Hint:** `cargo build` is already warning `fields id and text are never read` — that warning disappears once something outside the module can read them. Your Retriever will be that something.

### Warm-up B — stop ignoring `Result` in your store tests
`cargo build` prints seven `unused Result that must be used` warnings, all from `store.rs` — every `store.add(...)` in your tests throws away the `Result`. Add `.unwrap()` to each `add` call in the tests.

> **Hint:** Ask yourself *why* Rust warns here. If a bug ever made `add` reject a valid document, would your current tests notice? A `Result` you don't check is a test that can't fail.

### Warm-up C — one typo
`StoreError::DimensionMismatch`'s message says `"dimentions"`. Fix it to `"dimensions"`.

Run `cargo test -p agent-core` — still green, and those warnings should be gone.

---

## The Retriever

### Task 1 — a `RetrieveError` that wraps both failures
A retrieve does two things that can fail: embedding the query (`EmbedError`) and adding to the store (`StoreError`). Define a `RetrieveError` enum in `rag/mod.rs` with a variant for each, next to your other errors.

> **Hint:** `thiserror` can convert automatically. Look up `#[from]`:
> ```rust
> #[error("embedding failed: {0}")]
> Embed(#[from] EmbedError),
> ```
> Once you write `#[from]`, the `?` operator will turn an `EmbedError` into a `RetrieveError` for you — no `.map_err` needed.

### Task 2 — the `Retriever` struct
It owns an embedder and a store. The embedder must be a **trait object** because `Retriever` shouldn't care whether it's Ollama or something else:
```rust
pub struct Retriever {
    embedder: Box<dyn Embedder>,
    store: VectorStore,
}
```
Give it a `new(embedder: Box<dyn Embedder>) -> Self` that starts with an empty store.

> **Hint:** Why `Box<dyn Embedder>` and not `OllamaEmbedder`? Same reason `Agent` holds `Box<dyn Memory>` — so you can swap a fake embedder in tests without a network. You'll rely on exactly that in Task 5.

### Task 3 — `index`: embed a document and store it
```rust
pub async fn index(&mut self, id: &str, text: &str) -> Result<(), RetrieveError>
```
Embed the `text`, build a `Document`, and `add` it to the store.

> **Hint:** Two `?` in a row — one after `embed(...).await`, one after `add(...)`. If you did Task 1 with `#[from]`, both errors flow up automatically.
>
> **Hint:** Why `&mut self` here but `&self` in Task 4? Same question you answered for `VectorStore::add`.

### Task 4 — `retrieve`: the actual RAG step
```rust
pub async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, RetrieveError>
```
Embed the `query`, then call `self.store.search(...)` with the result.

> **Hint:** `search` returns a plain `Vec`, not a `Result` — so there's no `?` on that line. Only the `embed` call can fail here.

### Task 5 — test it with a *fake* embedder (no network!)
This is the important one. You do **not** want wiremock here — that tests Ollama, which you already tested in Stage 5. Here you're testing *retrieval logic*, so give it a predictable embedder.

Write a tiny fake in your test module:
```rust
struct FakeEmbedder;

#[async_trait::async_trait]
impl Embedder for FakeEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        // return a fixed vector per input — e.g. match on the text,
        // or hash the first char — anything deterministic
    }
}
```
Then write two tests:
1. `retrieve_returns_most_similar` — `index` two or three documents whose fake embeddings you control, `retrieve` a query, assert the closest one comes back first.
2. `retrieve_on_empty_index_returns_empty` — retrieve before indexing anything, expect an empty `Vec`.

> **Hint:** Make `FakeEmbedder` map text to vectors *you* can reason about geometrically, exactly like your Stage 6 search test. E.g. text containing "cat" → `[1.0, 0.0]`, text containing "dog" → `[0.0, 1.0]`. Then a "cat" query obviously retrieves the "cat" document.
>
> **Hint:** Because the embedder is fake and synchronous-in-spirit, these tests are `#[tokio::test]` (they `.await`) but run instantly and never touch the network.

### Done when
- [ ] `cargo test -p agent-core retriever` — green
- [ ] `cargo build` — no new warnings, and the seven Stage-6 warnings are gone
- [ ] You did **not** import `wiremock` in this file

### Bonus (only if the rest is done)
Right now `retrieve` embeds the query on every call, which is correct. But what if a caller retrieves the same query twice? Don't build a cache yet — just write a 2–3 line comment in `retrieve` describing *where* a cache would go and what its key would be. Naming the seam is the skill; building it is later.

---

**Why this matters:** after this stage, the flow `text in → most relevant stored text out` exists end-to-end. Stage 8 is the payoff — calling `retrieve` inside `Agent::run` and pasting the results into the system prompt so the LLM answers from *your* documents. Everything you've built (embedder, cosine similarity, store, retriever) converges there.
