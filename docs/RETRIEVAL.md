# Retrieval details

## Indexing
1. **Files:** `.md`, `.markdown` and `.txt` under `data.docs_dir` (and synced OneDrive files, if enabled).
   - Skipped: notes over `data.max_file_mb` (default 2 MB).
   - Skipped folders (`data.skip.dirs`, case-insensitive): `.git`, `.hg`, `.svn`, `.trash`, `trash`, `.trashed`,
     `$recycle.bin`, `.obsidian`, `.logseq`, `.vscode`, `.idea`, `.foam`, `.dendron`, `node_modules`, `__pycache__`.
   - Skipped file names (`data.skip.files`, globs): `*.bak`, `*.bak.*`, `*~`, `.#*`, `*.swp`, `*backup*`.
     The `backup` rule applies to **file names** only. A **folder** named `backup`/`backups` is demoted, not skipped (below).
2. **Front matter** (a leading `---` ... `---` block) is removed. Its `title:` is used as the note title if present,
   else the first `#` heading, else the file name.
3. **Sections:** the note is split at every Markdown heading outside code fences. Each section keeps its heading
   path (e.g. `Beds > North bed`).
4. **Windows:** each section is cut into chunks of at most `data.chunking.size` chars (default 800) with
   `data.chunking.overlap` chars of overlap (default 150). A window ends at the last newline in its back half, else
   the last space in its back half, else a hard cut. One long paragraph with no breaks still yields chunks within the limit.
5. **What gets indexed:** the dense model embeds `title > heading path` + newline + chunk. The embedding model's
   card (granite-embedding-english-r2) specifies no query/document prefixes, so none are used. A custom embedder can
   set `query_prefix` / `document_prefix` if its own card requires them. Vectors are L2-normalised. BM25 (SQLite FTS5)
   indexes title, heading path and chunk text as separate columns.
6. **Safety caps:** embedding and reranker inputs are truncated to `max_tokens` (default 512). That uses the
   model's own tokenizer from your local export, or a conservative character cap if it isn't available. The
   embedding model itself accepts 8192 tokens, and long inputs can exhaust RAM on CPU. Indexing embeds
   `dense.batch_size` (default 32) chunks at a time into one preallocated matrix.

Changing chunking or skip settings, `max_tokens`, or the model rebuilds the index on the next `plm-kit index` / `ask`.

## Ranking
1. BM25 and dense candidate lists, fused with RRF (k=60).
2. **Stale-copy demotion** (`retrieval.stale`): a score multiplier per chunk, from its note's path, applied **after
   fusion** (and inside the rerank blend). Nothing is dropped or deduplicated.

   | Rule | Matches (case-insensitive) | Default multiplier |
   |---|---|---|
   | archive | a **folder** segment exactly `archive`, `archived`, `backup`, `backups` or `old` (`archive_dirs`) | x0.5 (`archive_weight`) |
   | derived | any folder or file name matching `derived_patterns`: `*-export`, `*-exports`, `*_export`, `*_exports`, `*.html.md`, `*.html.txt`, `*chat-summary*`, `*chat-summaries*`, `*chat-export*` | x0.6 (`derived_weight`) |
   | owner | any folder or file name matching `owner_patterns` (empty by default; add your own authoritative folders) | x1.2 (`owner_weight`) |

   **Precedence:** derived and owner never combine. If both match, **owner wins**. The **archive rule stacks**
   (multiplies) with either one, e.g. `old/notes-export/x.md` = 0.5 x 0.6 = 0.3 and `archive/my-notes/x.md` = 0.5 x 1.2.
   The built-in derived list is deliberately conservative: it matches obvious machine-made copies only.

   **Your own lists:** each of owner/derived accepts globs (`owner_patterns`, `derived_patterns`: matched against
   each folder or file name) and regexes (`owner_regex`, `derived_regex`: searched in the whole lowercase relative
   path). Put them in `retrieval.stale`, or in a separate YAML/TOML file named by `retrieval.stale.lists_file`. The
   file's lists extend the config's; only those four keys are allowed. Start from `config/stale.example.yaml`
   (fictional examples) and keep your copy outside the repo. Owner lists are empty by default.
3. **Newest-copy step** (`retrieval.dedupe`, **off by default**). See below.
4. **Per-note cap** (`per_note_cap`, default 2): at most 2 chunks from one note in the results, so a long note can't
   fill every slot. `0` turns the cap off.
5. **Dense floor** (`dense_floor`, default off): if set, results whose dense cosine is below it are removed. It's
   off because the synthetic tests don't justify a value, and on real notes the cosine scale is poorly separated
   (see the gate below). Try it only after calibrating on your own questions.
6. Optional reranker (`fused+rerank`, or `--preset quality`).
7. **Not-in-notes gate** (`not_in_notes_gate`: `block` default, `warn`, `off`; `not_in_notes_threshold` 0.78):
   if the best dense cosine is below the threshold, `block` answers "Not in your notes", `warn` shows the results
   with a "weak match" warning, and `off` ignores it. **It is unreliable on real notes.** In the maintainer's
   re-run, out-of-notes questions scored 0.775 / 0.851 / 0.897 (min / median / max) and answerable ones had min
   0.770 and median 0.889. At 0.78 the gate caught 1 of 8 out-of-notes questions with 1 false block; at 0.82, 3
   caught for 4 false blocks; at 0.84, 4 caught for 11 false blocks. No single cutoff separates them.
   `plm-kit calibrate` on your own questions shows your trade-off (catches vs false blocks at the current and
   suggested threshold, and whether the scores overlap). It picks a trade-off; it can't fix the overlap. We
   suggest `warn` or `off`.

Dense-only mode (`mode: dense`) returns the raw dense order with the per-note cap. Stale demotion applies to the
fused modes.

## Newest-copy step (experimental, off by default)
**Recommendation:** leave it off. If you try it, use `demote` only. `drop` is kept for experiments but is **not
recommended**.

> **Warning: it trusts file modification times.** Synced or copied folders (OneDrive, Dropbox, git checkouts,
> restores from backup) reset them, often to one bulk time for many files, so "newest" may mean nothing. In the
> maintainer's real-notes test it didn't help (MRR slightly lower with it on), some groups had tied times, the one
> expected note inside a group was the *older* copy, and `demote` once pushed the right answer out of the top 10.

Settings: `retrieval.dedupe.newest_copy` = `off` (default) | `demote` | `drop`, `threshold` (0.9), `multiplier`
(0.5), `shingle_words` (5), `min_words` (20), `tie_tolerance_s` (60).
- **Grouping** (when the retriever is built): two notes are near-duplicates when their **normalized titles** match
  (lowercased words, trailing copy markers like "copy", "old", "final", "v2" or a number removed) **and** the
  Jaccard similarity of their 5-word shingle sets is at least `threshold`. Groups are the connected components.
  Notes under `min_words` are never grouped. Only notes with the same normalized title are compared, so the cost
  stays small.
- **At query time**, after fusion and stale demotion: if two or more notes of one group are among the candidates,
  the newest of them keeps its score **only if it is more than `tie_tolerance_s` seconds newer** than the next one.
  The others are then multiplied by `multiplier` (`demote`) or removed (`drop`). Tied or near-tied times mean no
  winner: the group is left alone (there is no name-based tiebreak). An unknown time counts as oldest. A group
  member retrieved on its own is untouched. Dense-only mode doesn't apply it.
- **Reporting:** `plm-kit index` and `plm-kit eval` print `newest_copy: {mode, groups, notes_in_groups,
  largest_group, groups_without_clear_newest, tie_tolerance_s}`. Many groups without a clear newest copy is a sign
  your file times came from a sync or copy. `plm-kit health` shows the configured mode.
- **Other risks:** a shared template or boilerplate can make unrelated notes look like copies; the newer note
  isn't always the live one (e.g. a scratch copy edited after the real note); `drop` can hide the only copy that
  holds an answer.
- **Compare on vs off** before enabling it: `plm-kit eval --docs DIR --gold FILE --newest-copy off --json off.json`,
  then the same with `--newest-copy demote`; `--dedupe-threshold`, `--dedupe-multiplier` and
  `--dedupe-tie-tolerance` try other values.

## Known failure cases
- A **live note inside a folder named e.g. `old`** (or `archive`) gets demoted. Rename the folder, or remove the
  name from `archive_dirs`.
- A **renamed stale copy stored elsewhere** (e.g. `notes/wifi copy.md`) isn't caught by stale demotion, which matches by path only. The opt-in newest-copy step can catch it if the copy is the older one.
- **Live and stale copies can both take result slots.** Demotion lowers scores; it doesn't remove duplicates.
- **Stacked weights can bury a note that is the only source** for an answer (e.g. the only copy lives in
  `archive/chat-exports/`). Lower the weights or add the folder to `owner_patterns`.
- Path globs can over-match: a folder you named `my-export` for your own writing counts as derived.

## Possible future work (not implemented)
- Alias / synonym expansion of queries.
- A boost for bold-term definitions (`**term**: ...`).
- Per-source-type weights (e.g. competitor or reference cards).
- Content-based stale-copy detection (near-duplicate text) instead of path rules.
