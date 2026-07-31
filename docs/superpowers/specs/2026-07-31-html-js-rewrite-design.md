# IDBac Interactive Interface — HTML + JS Rewrite

**Date:** 2026-07-31
**Status:** Design, pending review
**Replaces:** the Streamlit app rooted at `Protein_Dendrogram.py` + `pages/*.py` (~5,800 lines)

## Why

Streamlit re-executes the entire script on every widget interaction, over a stateful
websocket, inside a per-user server process. Every slider drag costs a server round trip
and a full re-render. That model does not survive a growing user base: concurrency is
bounded by server CPU and RAM, not by the (small) amount of work each user actually needs.

The data behind this app is read-only and per-task. Once a task is loaded there is no
reason for any subsequent interaction to touch the server at all.

## Goals

1. Preserve development quality — real abstractions, no copy-paste, testable units.
2. Results must be consistent with the Streamlit version.
3. Load all data once per session; every interaction after that is local and instant.
4. Stay visually and operationally cohesive with idbac.org (the IDBac-KB-Server Flask/Dash app).

## Architecture

Two tiers, split on one hard rule:

> **The server reshapes data. The browser does all of the science.**

Anything parameterised by a UI control lives in the browser. Anything that is a pure
function of the task ID lives on the server, where it is cached once and shared by every
user of that task.

```
┌─────────────────────────────────────────────────────────┐
│ Browser (static SPA, no build step)                      │
│                                                          │
│  TaskStore ──> pages ──> figures ──> Plotly.js           │
│     ▲            │                    vis-network        │
│     │            └──> numeric/ (linkage, PCA, PCoA, …)   │
│     │                                                    │
│  one fetch per session                                   │
└─────┼───────────────────────────────────────────────────┘
      │  GET /api/task/<id>/bundle
┌─────▼───────────────────────────────────────────────────┐
│ Flask "bundle" service (stateless, cache keyed on task)  │
│  fetch ~13 GNPS2 result files → parse → compact → cache  │
└─────┬───────────────────────────────────────────────────┘
      │
   gnps2.org/resultfile, idbac.org/api, NCBI E-utilities
```

### Why not pure static

`/resultfile` does send permissive CORS headers, so a browser *can* read GNPS2 task
files directly. One file makes that impractical:

| File | Size (task `4c43…`) | Size (task `8e0c…`) |
|---|---|---|
| `nf_output/search/db_db_distance.tsv` | **168 MB** | **77 MB** |
| everything else combined | ~5 MB | ~9 MB |

`db_db_distance.tsv` is the pairwise distance matrix between knowledgebase hits, stored
as text with both `(i,j)` and `(j,i)` rows and two redundant ID columns per row. Streamlit
gets away with it because `@st.cache_data` parses it once per server process and shares
it across all users.

Re-encoded as a float32 upper triangle plus an ID index, the same information is
**1.75 MB / 3.76 MB** — a 44× reduction, and that is the *worst case* (`max_db_results = -1`,
every KB hit shown). At the default setting only ~100 of 1.88 M pairs are ever read.

So the server's one indispensable job is transcoding. It is not doing science, and it takes
no user parameters, which is what keeps it cacheable and scalable.

A second, smaller reason: `status.json` does **not** send CORS headers (verified), and NCBI
E-utilities needs rate limiting. Both get proxied.

### Tier 1 — `api/` (Flask)

Mirrors IDBac-KB-Server's stack exactly so the two services are operationally identical:
Flask + gunicorn + Flask-Caching (filesystem) + nginx-proxy `VIRTUAL_HOST`.

| Route | Purpose |
|---|---|
| `GET /api/task/<id>/bundle` | The whole session payload: manifest + JSON parts |
| `GET /api/task/<id>/matrix/<name>` | Binary float32 payloads (`db_db`, `query_query`, `spectra`) |
| `GET /api/task/<id>/status` | CORS proxy for `status.json` |
| `GET /api/spectrum/peaks?usi=` | Proxy for metabolomics-usi + `idbac.org/api/spectrum/filtered` |
| `POST /api/genbank/enrich` | Batched, rate-limited NCBI lookup |

Cache key is the task ID alone. Entries are immutable (GNPS2 tasks are write-once), so
they can be cached indefinitely and served from disk.

The bundle also does the two per-task aggregations that carry **no** user parameters, so
they are computed once server-side rather than 100× in every browser:

- `small_molecule/summary.json` (5.7 MB of per-scan peaks) → per-file merged arrays
  (`m/z`, mean intensity, replicate frequency). This is `Small_Molecule_Utils.get_small_molecule_dict`
  moved verbatim; its rounding and averaging are fixed, so it is data shaping, not analysis.
  The *filters* over it stay in the browser.
- `complete_enriched_db_results.tsv` (3–5 MB) → deduplicated KB metadata table + a compact
  query→hit distance list.

### Tier 2 — `web/` (static SPA)

Vanilla ES modules, no build step, no `node_modules` in the repo. Vendored `plotly.js`,
`vis-network`, and Bootstrap 5. Rationale: the maintainers are a Python bioinformatics lab;
a build toolchain is a permanent tax, and "HTML + JS" was the explicit ask. Module
boundaries and JSDoc types carry the structure instead.

```
web/
  index.html                  shell: navbar, task input, page container
  css/idbac.css               design tokens shared with IDBac-KB-Server
  js/
    main.js                   router + session bootstrap
    core/
      store.js                TaskStore — the load-once session data
      bundle.js               fetch + decode bundle (incl. binary matrices)
      urlstate.js             shareable-link params (replaces st.query_params)
      format.js               label/column formatting
      parse.js                parse_numerical_input, bin <-> m/z helpers
    numeric/
      matrix.js               squareform, condensed indexing
      distance.js             cosine / euclidean / presence
      linkage.js              scipy-exact linkage
      dendrogram.js           scipy-exact dendrogram (ivl, icoord, dcoord, colors)
      newick.js  pca.js  pcoa.js  tsne.js
      layout.js               spring / circular / spectral / kamada-kawai
      community.js            greedy modularity
    figures/                  pure functions: data -> Plotly figure spec
    pages/                    one module per page
    ui/
      controls.js             declarative control panel (replaces st.slider/selectbox)
      expander.js  table.js  downloads.js  notify.js
```

`ui/controls.js` is the piece that earns its keep. Streamlit's real value was declarative
widgets, and losing that is how a rewrite like this turns into DOM spaghetti. Pages will
declare controls as data:

```js
{ type: 'slider', key: 'coloringThreshold', label: 'Colour code clusters…',
  min: 0, max: 1, step: 0.05, default: 0.6, help: '…' }
```

and get two-way binding to the store, URL-param sync, and re-render scheduling for free.

## Consistency (goal 2)

The riskiest part of this project, so it gets an explicit contract rather than good
intentions. Every computation is classified:

**Tier A — must match Python numerically (tolerance 1e-9).** All deterministic:

- `squareform`, `cosine_distances`, `euclidean_distances`
- `scipy.cluster.hierarchy.linkage` — nn-chain for `average`/`complete`/`weighted`/`ward`,
  MST for `single`, generic for `centroid`/`median`
- `scipy.cluster.hierarchy.dendrogram` — leaf order (`ivl`), `icoord`/`dcoord`,
  `color_list`, `leaves_color_list`, and `color_threshold` semantics
- Newick export, all heatmap filter pipelines, KB result filtering, small-molecule aggregation
- PCA (SVD + sklearn's `svd_flip` sign convention)
- PCoA (eigendecomposition of the double-centred −½·D²)

**Tier B — same algorithm, not bit-identical.** Stochastic or optimiser-dependent, and
never a reported number — layout coordinates only:

- t-SNE (random init; sklearn's own output already varies run to run, as the current app
  sets no seed)
- `kamada_kawai_layout` (scipy L-BFGS-B), greedy modularity community ordering
- `spring_layout` is seeded (42) and *is* reproducible, so it is held to Tier A structure

**A note on why this needs real care.** The app's own documentation says "a flat line
between strains at 0 represents identical spectra" — meaning exact zero distances, and
therefore *ties*, are normal here. Tied distances are where two "correct" linkage
implementations diverge in leaf order. Reimplementing scipy's algorithm faithfully
(including merge ordering and label conventions) is required; a merely correct
hierarchical clustering is not sufficient.

**Enforcement — differential test harness.** This is the mechanism that makes goal 2 a
checked property:

```
tests/
  golden/generate.py     drives Python reference impls over 2 real tasks -> golden JSON
  golden/*.json          committed fixtures
  numeric/*.test.js      node --test, asserts JS == golden
```

`node --test` ships with Node 20, so the test suite needs no dependencies either. Golden
fixtures are generated from the two real tasks already validated against this app:
`4c43a2ca6f3541938e491b3c52442721` (KB hits, no metadata, no small molecules) and
`8e0cb0c6a3c04ae1991bbc1dca2882b5` (metadata + small molecules).

## Rendering

Deliberately the same engines, so figures are ports rather than reinterpretations:

| Streamlit | Browser |
|---|---|
| `plotly` (Python) | `plotly.js` — same figure JSON, same renderer |
| `pyvis` | `vis-network` — pyvis is a thin wrapper over it |
| `st.*` widgets | `ui/controls.js` |

Figure code becomes pure `data -> figure spec` functions, which makes it both testable
and reusable across the protein and small-molecule heatmaps (currently duplicated).

## Cohesion with idbac.org

Lifted from `IDBac-KB-Server/assets/styles.css` into a shared `idbac.css`:

- Primary blue `#156082` (hover `#1a7ca1`), grey `#3a3a3a`, orange accent `#d88000`
- Squared buttons (`border-radius: 1px`), `.grey-box` `#f0f0f0` / `#6f6f6f`
- `max-width: 1600px`, white background, Bootstrap 5 base
- Sticky `GNPS2xIDBac.png` navbar with the right-aligned Sign Up button
- The same Umami analytics tag (`4611e28d-c0ff-469d-a2f9-a0b54c0c8ee0`)

## Cleanups folded in

Duplication found while reading the current code, resolved by the module layout above
rather than by separate refactoring:

- `get_peaks_a` / `get_peaks_b` (`01_Plot_Spectra.py`) are byte-identical, 59 lines each
- `basic_dendrogram` is near-duplicated in `02_Protein_Heatmap.py` and `04_Metabolite_Association_Network.py`
- `draw_protein_heatmap` exists twice (`Protein_Dendrogram_Components.py`, largely dead, and `02_Protein_Heatmap.py`)
- `DEV-`/`BETA-` base-URL derivation is repeated in six places
- Heatmap + dendrogram-overlay rendering is duplicated between the protein and small-molecule pages
- Leaf-colour → cluster-ID assignment is duplicated
- Two large `if False:` dead blocks in `Protein_Dendrogram_Components.py`

## Scope

All seven pages are ported: Dendrogram (the entry page), Plot Spectra, Protein Heatmap,
Dimensionality Reduction, Metabolite Association Network, Search Results, Deposition QC.

Deposition QC is a special case worth flagging: it reads a *different* task ID space
(deposition dry runs, not analysis tasks) and its own files, and the page itself states it
is non-functional for users outside the KB deposition workflow. It is ported as-is — a
table plus rendered spectrum images — and does not share the task bundle.

The Streamlit app is left in place during the port so the two can be compared side by
side; removing it is a follow-up once parity is confirmed.

## Risks

| Risk | Mitigation |
|---|---|
| Tied distances → divergent leaf order | Port scipy's exact algorithm; golden tests over real tasks |
| A task's `db_db_distance.tsv` is far larger than 168 MB | Server streams and filters row-wise, never loads the TSV whole |
| Bundle cache grows unbounded | LRU on disk, keyed by task ID; entries are immutable |
| t-SNE cannot match | Documented as Tier B; add an optional seed so it is at least reproducible in the new app |
| Plotly.js and Python Plotly drift | Pin the vendored version; figure specs are compared in golden tests |
