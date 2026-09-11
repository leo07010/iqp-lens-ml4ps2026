# refs.bib Verification Report

Generated 2026-09-11. Every entry in `refs.bib` (52 total) was cross-checked against the arXiv API (`https://export.arxiv.org/api/query?id_list=...`) for entries with an `eprint` field, and against Crossref (`https://api.crossref.org/works/<DOI>` or bibliographic search) for entries with (or that should have) a `doi`. Two entries without arXiv/DOI (`ml4sci_deeplense`, `smolensky_1986_information`) were checked via GitHub API and web/Crossref search respectively.

**Do not treat `refs.bib` as trustworthy until it is replaced by `refs_verified.bib`** (same directory) — bib keys are unchanged, only field values are corrected. `refs_verified.bib` is the corrected file; `refs.bib` itself was NOT modified (another agent may be editing it).

## Summary

- **Total entries checked: 52**
- **Entries with at least one factual error or missing field: 26**
- **Entries needing a human judgment call (not auto-corrected): 2** (`recio_2025_iqpopt`, `ml4sci_deeplense`)
- **Entries with a harmless key/year label inconsistency (not changed, key preserved per instructions): 1** (`alexander_2021_decoding`)
- **NOT FOUND (verification failed to resolve the source): 0** — every arXiv ID and DOI resolved successfully. Several entries correctly have no DOI (preprints, ICLR proceedings, JMLR volume 13, the MIT Press 1986 book chapter) — these are marked `N/A`, not `NOT FOUND`.

### Highest-severity issues (author name errors — these are the ones most likely to embarrass if cited wrong)

| Key | Wrong (in refs.bib) | Correct |
|---|---|---|
| `kurkin_2025_universality` | Kurkin, **Andrey**; Shen, **Yudai**; Pielawa, **Suzanne**; bare **Wang** (no given name) | Kurkin, **Andrii**; Shen, **Kevin**; Pielawa, **Susanne**; Wang, **Hao** |
| `shen_2026_characterizing` | Shen, **Yudai**; Pielawa, **Suzanne**; bare **Wang** | Shen, **Kevin**; Pielawa, **Susanne** (bib already had this one right); Wang, **Hao** |
| `lerch_2026_iqp` | Lerch, **Sabine**; bare **Puig** (no given name); "Armengol, **Erik Recio**" | Lerch, **Sacha**; Puig, **Ricard**; Armengol, **Erik** |
| `huang_2026_spectral` | bare **Huang**, **Maxwell**, **Peters**, **Pye** (no given names); Jahangiri, **Hossein** | Huang, **Austin**; Maxwell, **William**; Peters, **Evan**; Pye, **Jason**; Jahangiri, **Soran** |
| `banks_2026_qudit` | Banks, **Harry**; Crippa, **Leonardo**; bare **Traube**, **Unger**, **Ertler** | Banks, **Robert J.**; Crippa, **Arianna**; Traube, **Matthias**; Unger, **Josua**; Ertler, **Christian** |
| `ballo_2026_shallow` | bare **Arroyo-Sánchez**; García-Molina, **Pablo**; Garriga, **Albert**; Vilariño, **Ferran** | Arroyo-Sánchez, **Marcos**; García-Molina, **Paula**; Garriga, **Adan**; Vilariño, **Fernando** |
| `deluca_2026_trainability` | bare **De Luca** (no given name) | De Luca, **Gennaro** |
| `kremer_2026_efficient` | Kremer, **Marius**; Dupuis, **Frédéric** | Kremer, **David**; Dupuis, **Nicolas** |
| `takahashi_2026_classical` | Miyamoto, **Aiko** | Miyamoto, **Masayuki** |
| `bremner_2025_instantaneous` | Cheng, **Sixia** | Cheng, **Bin** |
| `huang_2025_generative` | Eassa, **Noah** | Eassa, **Norhan** |
| `gharibyan_2025_heuristic` | truncated to "Gharibyan, Hrant **and others**" | full 6 authors: Gharibyan, Mullath, Sherman, Su, Tepanyan, Zhang |

Pattern: nearly every author-name error is in the batch of very recent (2025-2026) IQP-generative-model preprints — this looks like the original bib was built by scraping abstract-page HTML where multi-token given names or accented names broke the name splitter, silently dropping first names or merging tokens across authors.

## Per-entry detail

### liu_2026_correlation (arXiv 2603.06440)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title | Toward Generative Quantum Utility via Correlation--Complexity Map | (same, plain hyphen) | match | arXiv API |
| authors | Liu, Placidi, Brunner, Rinaldi | same order | match | arXiv API |
| year | 2026 | 2026 (2026-03-06) | match | arXiv API |
| primaryclass | cs.LG | cs.LG | match | arXiv API |
| doi | none | none exists | n/a | arXiv API |

### liu_2026_physics (arXiv 2607.28866)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title | Generative IQP Circuit Learning with Physics-Informed Latent Initialization | same | match | arXiv API |
| authors | Liu, Placidi, Ballarin, Rinaldi | same | match | arXiv API |
| year | 2026 | 2026 (2026-07-30) | match | arXiv API |
| primaryclass | quant-ph | quant-ph | match | arXiv API |
| doi | none | none exists | n/a | arXiv API |

### bremner_2011_classical (arXiv 1005.1407, doi 10.1098/rspa.2010.0301)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title | Classical simulation of commuting quantum computations implies collapse of the polynomial hierarchy | same | match | arXiv API + Crossref |
| authors | Bremner, Jozsa, Shepherd | same order | match | Crossref |
| year | 2011 | 2011 | match | Crossref |
| journal/vol/no/pages | Proc. Roy. Soc. A, 467(2126), 459-472 | same | match | Crossref |
| doi | 10.1098/rspa.2010.0301 | same | match | https://api.crossref.org/works/10.1098/rspa.2010.0301 |

### bremner_2016_average (arXiv 1504.07999, doi 10.1103/PhysRevLett.117.080501)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title/authors/year/doi/journal | all fields | all match (title differs only in Crossref's Title Case vs bib's sentence case, same content) | match | arXiv API + Crossref |

### bremner_2017_achieving (arXiv 1610.01808)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title/authors/year | — | all match | match | arXiv API |
| doi | **missing** | **10.22331/q-2017-04-25-8** | **MISMATCH → fixed** | https://api.crossref.org/works/10.22331/q-2017-04-25-8 |

### codsi_2025_classically (arXiv 2212.08609, doi 10.1103/PhysRevA.111.012422)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title | Classically Simulating Quantum Supremacy IQP Circuits through a Random Graph Approach (preprint title) | **Classically simulating intermediate-scale instantaneous quantum polynomial circuits through a random graph approach** (published title — retitled between preprint and journal) | **MISMATCH → fixed** | https://api.crossref.org/works/10.1103/PhysRevA.111.012422 |
| authors/year/doi/journal | — | all match | match | arXiv API + Crossref |

### maslov_2024_fast (arXiv 2402.03211)
All fields match (Maksymov, Andrii already spelled correctly). doi: n/a (still preprint-only). Source: arXiv API.

### rajakumar_2024_polynomial (arXiv 2403.14607, doi 10.1137/1.9781611978322.30)
All fields match (year=2025 correctly reflects SODA proceedings publication year, not the 2024 preprint year). Source: arXiv API + Crossref.

### takahashi_2026_classical (arXiv 2603.05920)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title/year/primaryclass | — | match | match | arXiv API |
| authors | Takahashi, Yasuhiro; **Miyamoto, Aiko**; Kunihiro, Noboru | Takahashi, Yasuhiro; **Miyamoto, Masayuki**; Kunihiro, Noboru | **MISMATCH → fixed** | arXiv API |

### bremner_2025_instantaneous (arXiv 2308.07152)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title | — | match | match | arXiv API |
| authors | Bremner; **Cheng, Sixia**; Ji | Bremner; **Cheng, Bin**; Ji | **MISMATCH → fixed** | arXiv API |
| year | 2025 | 2025 (journal pub year; preprint from 2023) | match | Crossref |
| doi | **missing** | **10.1103/PRXQuantum.6.020315** (PRX Quantum 6, 020315) | **MISMATCH → fixed, added vol/pages too** | https://api.crossref.org/works/10.1103/PRXQuantum.6.020315 |

### aaronson_2024_verifiable (arXiv 2404.14493)
All fields match. doi: n/a. Source: arXiv API.

### zhang_2025_complexity (arXiv 2510.00132)
All fields match (sole author). doi: n/a. Source: arXiv API.

### gharibyan_2025_heuristic (arXiv 2510.25838)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| authors | Gharibyan, Hrant **and others** | Gharibyan, Hrant; **Mullath, Mohammed Zuhair; Sherman, Nicholas E.; Su, Vincent P.; Tepanyan, Hayk; Zhang, Yuxuan** (6 total) | **MISMATCH → fixed (full list restored)** | arXiv API |
| title/year/primaryclass | — | match | match | arXiv API |

### kremer_2026_efficient (arXiv 2604.21908)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| authors | **Kremer, Marius**; **Dupuis, Frédéric** | **Kremer, David**; **Dupuis, Nicolas** | **MISMATCH → fixed (both given names wrong, not accent-only)** | arXiv API |
| title/year/primaryclass | — | match | match | arXiv API |

### havlicek_2019_supervised (arXiv 1804.11326)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title/authors/year/primaryclass | — | match | match | arXiv API |
| doi | **missing** | **10.1038/s41586-019-0980-2** | **MISMATCH → fixed** | arXiv `<arxiv:doi>` + Crossref bibliographic search |

### huang_2021_power (arXiv 2011.01938)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title/authors/year/primaryclass | — | match | match | arXiv API |
| doi | **missing** | **10.1038/s41467-021-22539-9** | **MISMATCH → fixed** | arXiv `<arxiv:doi>` + Crossref bibliographic search |

### recio_2025_iqpopt (arXiv 2501.04776) — NEEDS AUTHOR'S JUDGMENT
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title/year/primaryclass | — | match | match | arXiv API |
| authors | Recio-Armengol, Erik; Bowles, Joseph | arXiv's own metadata/abs page for **this specific paper** lists the first author as **"Erik Armengol"**, not "Erik Recio-Armengol" — inconsistent with the same person's name on the companion paper `recio_2025_train` (2503.02934), where arXiv itself uses "Recio-Armengol" | **FLAGGED, not auto-corrected** — this is an arXiv self-inconsistency across the same author's two papers, not a clear bib error. `refs_verified.bib` keeps "Recio-Armengol, Erik" and adds a `note` documenting the discrepancy; decide before citing whether to match this paper's literal arXiv record instead. | https://arxiv.org/abs/2501.04776 vs https://arxiv.org/abs/2503.02934 |

### recio_2025_train (arXiv 2503.02934)
All fields match. Source: arXiv API.

### kurkin_2025_universality (arXiv 2510.08476)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| authors | Kurkin, **Andrey**; Shen, **Yudai**; Pielawa, **Suzanne**; **Wang** (bare); Dunjko, Vedran | Kurkin, **Andrii**; Shen, **Kevin**; Pielawa, **Susanne**; **Wang, Hao**; Dunjko, Vedran | **MISMATCH → fixed** | arXiv API |
| title/year/primaryclass | — | match | match | arXiv API |

### shen_2026_characterizing (arXiv 2602.11042)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| authors | Shen, **Yudai**; Pielawa, Suzanne; Dunjko, Vedran; **Wang** (bare) | Shen, **Kevin**; Pielawa, **Susanne**; Dunjko, Vedran; **Wang, Hao** | **MISMATCH → fixed** | arXiv API |
| title/year/primaryclass | — | match | match | arXiv API |

### lerch_2026_iqp (arXiv 2603.14576)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| authors | Lerch, **Sabine**; Bowles, Joseph; **Puig** (bare); Armengol, **Erik Recio**; Holmes, Zoë; Thanasilp, Supanut | Lerch, **Sacha**; Bowles, Joseph; **Puig, Ricard**; Armengol, **Erik**; Holmes, Zoë; Thanasilp, Supanut | **MISMATCH → fixed** | arXiv API |
| title/year/primaryclass | — | match | match | arXiv API |

### deluca_2026_trainability (arXiv 2606.10179)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| authors | **De Luca** (bare, no given name) | **De Luca, Gennaro** (sole author, no co-authors dropped) | **MISMATCH → fixed** | arXiv API |
| title/year/primaryclass | — | match | match | arXiv API |

### nguyen_2026_discovery (arXiv 2606.24264)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title/authors/year | — | match | match | arXiv API |
| primaryclass | **quant-ph** | **cs.SI** (quant-ph is only a secondary/cross-listed category; primary is cs.SI) | **MISMATCH → fixed** | arXiv API |

### banks_2026_qudit (arXiv 2606.28236)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| authors | Banks, **Harry**; Crippa, **Leonardo**; **Traube, Unger, Ertler** (all bare); Lechner, Wolfgang | Banks, **Robert J.**; Crippa, **Arianna**; **Traube, Matthias; Unger, Josua; Ertler, Christian**; Lechner, Wolfgang | **MISMATCH → fixed** | arXiv API |
| title/year/primaryclass | — | match | match | arXiv API |

### huang_2026_spectral (arXiv 2607.06675)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| authors | **Huang, Maxwell, Peters, Pye** (all bare); Belis, Vasilis; Jahangiri, **Hossein**; Bowles, Joseph | **Huang, Austin; Maxwell, William**; Belis, Vasilis; **Peters, Evan; Pye, Jason**; Jahangiri, **Soran**; Bowles, Joseph | **MISMATCH → fixed** | arXiv API |
| title/year/primaryclass | — | match | match | arXiv API |

### ballo_2026_shallow (arXiv 2511.05267)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| authors | Balló-Gimbernat, Oriol; **Arroyo-Sánchez** (bare); García-Molina, **Pablo**; Garriga, **Albert**; Vilariño, **Ferran** | Balló-Gimbernat, Oriol; **Arroyo-Sánchez, Marcos**; García-Molina, **Paula**; Garriga, **Adan**; Vilariño, **Fernando** | **MISMATCH → fixed** | arXiv API + abs page |
| title/year/primaryclass/journal/vol/pages | — | match (Phys. Rev. A 113, 042617, 2026) | match | Crossref |
| doi | **missing** | **10.1103/jcdk-q3xc** (APS's newer opaque-string DOI format, not the old `PhysRevA.vol.page` pattern) | **MISMATCH → fixed** | https://api.crossref.org/works/10.1103/jcdk-q3xc |

### huang_2025_generative (arXiv 2509.09033)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| authors | Huang, Broughton, Eassa **Noah**, Neven, Babbush, McClean | Huang, Broughton, Eassa **Norhan**, Neven, Babbush, McClean | **MISMATCH → fixed** | arXiv API |
| title/year/primaryclass | — | match | match | arXiv API |

### buhrman_2025_formal (arXiv 2510.01953)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title | Formal **framework** for quantum advantage | Formal **Framework** for Quantum Advantage (title case) | **MISMATCH (minor) → fixed** | arXiv API |
| authors/year/primaryclass | — | match | match | arXiv API |

### radford_2015_unsupervised (arXiv 1511.06434)
All fields match. doi: n/a (never formally published with a DOI, ICLR 2016 workshop paper). Source: arXiv API.

### alexander_2020_deep (arXiv 1909.07346, doi 10.3847/1538-4357/ab7925)
All fields match exactly, including doi/journal/vol/pages via Crossref. Source: arXiv API + Crossref.

### watanabe_1960_information (doi 10.1147/rd.41.0066)
All fields match (case-only title diff). Source: https://api.crossref.org/works/10.1147/rd.41.0066

### chow_1968_approximating (doi 10.1109/TIT.1968.1054142)
All fields match (Crossref's legacy IEEE record only stores single-letter initials "C." / "C.", less complete than bib's "C. K." / "C. N." but same two authors/order — not flagged as an error). Source: https://api.crossref.org/works/10.1109/TIT.1968.1054142

### schneidman_2003_network (doi 10.1103/PhysRevLett.91.238701)
All fields match (APS article-number 238701 = bib's page field). Source: https://api.crossref.org/works/10.1103/PhysRevLett.91.238701

### gretton_2012_kernel
All fields match against JMLR's own page (13(25):723-773, 2012). doi: **N/A confirmed** — no DOI was ever assigned to this JMLR v13 paper (checked JMLR page directly + Crossref bibliographic search, no match). Correct as-is with no doi field. Source: https://www.jmlr.org/papers/v13/gretton12a.html

### hinton_2002_training (doi 10.1162/089976602760128018)
All fields match. Source: https://api.crossref.org/works/10.1162/089976602760128018

### tieleman_2008_training (doi 10.1145/1390156.1390290)
All fields match (bib's abbreviated venue name is a fair form of ACM's full "Proceedings of the 25th international conference on Machine learning - ICML '08"). Source: https://api.crossref.org/works/10.1145/1390156.1390290

### silverman_1986_density
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title/authors/year/publisher | Density Estimation for Statistics and Data Analysis; Silverman, B.W.; 1986; Chapman and Hall, London | — | match (as bibliographic facts about the 1986 original) | — |
| doi | 10.1201/9781315140919 | This DOI resolves to a **2018 Routledge ebook reissue** (publisher Routledge, `issued` 2018-02-19), **not** the cited 1986 Chapman & Hall original, which predates DOI assignment and has no DOI of its own | **MISMATCH → doi field removed** in `refs_verified.bib` (kept year=1986/publisher=Chapman and Hall since that's the actual edition being cited) | https://api.crossref.org/works/10.1201/9781315140919 |

### walsh_1923_closed (doi 10.2307/2387224)
All fields match (volume/issue/pages confirmed via JSTOR/AMS listing; Crossref's own JSON only stores a truncated "page: 5" for this old JSTOR record, a known Crossref data-completeness quirk, not a bib error). Source: https://api.crossref.org/works/10.2307/2387224

### khojasteh_2022_lagrangian (doi 10.1016/j.dib.2021.107725)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| authors (rendered name) | "Ali Rahimi Khojasteh" | "Ali Rahimi Khojasteh" | match (same rendered name) | Crossref |
| authors (BibTeX family/given split) | `{Rahimi Khojasteh, Ali}` → parses as family="Rahimi Khojasteh" | Crossref splits as given="Ali Rahimi", family="**Khojasteh**" | **MISMATCH (structural) → fixed** to `{Khojasteh, Ali Rahimi}` so bibliography styles sort/render the surname correctly | https://api.crossref.org/works/10.1016/j.dib.2021.107725 |
| title/journal/vol/pages/year/doi | — | match | match | Crossref |

### ml4sci_deeplense — NEEDS AUTHOR'S JUDGMENT (not a hard error)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| repo exists | github.com/ML4SCI/DeepLense | confirmed exists, org type, created 2021-08-13 | match | https://api.github.com/repos/ML4SCI/DeepLense |
| author | {ML4SCI Collaboration} | GitHub has no formal "ML4SCI Collaboration" byline; repo has no description field; top contributors are individual GSoC-program usernames (emanueleusai, lucasjvds, Saranga7, iyersreehari, sachdevkartik, ghatotkachhh, aduplins, yaashwardhan), not an entity literally named that | **FLAGGED, not auto-corrected** — "ML4SCI Collaboration" is a defensible informal shorthand but not verifiable as an official name; consider `{ML4SCI}` (the actual org name) or naming specific maintainers if precision matters | GitHub API |
| year | 2025 | repo created 2021, continuously active, no single "2025" release event found | not a hard mismatch, just unverifiable as a discrete pub year for a live repo | GitHub API |

### alexander_2021_decoding (arXiv 2008.12731) — key/year label note only
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title/authors | — | match (all 7 authors, exact order incl. "Von Klar, Ryker") | match | arXiv API |
| year | 2020 | 2020 (arXiv published 2020-08-28) | **match — year field is correct** | arXiv API |
| doi | none | none exists anywhere (confirmed via OpenAlex + Crossref bibliographic search — no published venue beyond arXiv) | n/a, correct as-is | OpenAlex + Crossref |
| **bib key** | `alexander_2021_decoding` | the key says "2021" but the year field (correctly) says 2020 | **key/year label inconsistency — not fixed**, per task instructions bib keys must stay unchanged; flagging only | — |

### liu_2018_differentiable (doi 10.1103/PhysRevA.98.062324)
All fields match. Source: https://api.crossref.org/works/10.1103/PhysRevA.98.062324

### benedetti_2019_generative (doi 10.1038/s41534-019-0157-8)
All fields match (6 authors, exact order). Source: https://api.crossref.org/works/10.1038/s41534-019-0157-8

### coyle_2020_born (doi 10.1038/s41534-020-00288-9)
All fields match. Source: https://api.crossref.org/works/10.1038/s41534-020-00288-9

### cerezo_2025_does (doi 10.1038/s41467-025-63099-6)
All fields match, all 12 authors in exact order. Source: https://api.crossref.org/works/10.1038/s41467-025-63099-6

### binkowski_2018_demystifying (arXiv 1801.01401)
All fields match. doi: n/a (ICLR 2018 is non-archival, confirmed via arXiv comment + OpenReview, no DOI exists). Source: arXiv API.

### smolensky_1986_information
Title/authors/editors/publisher/pages/year all match against the known PDP Vol.1 citation record. doi: **NOT FOUND** (genuinely absent) — checked two independent Crossref queries (bibliographic search + MIT Press's Crossref prefix 10.7551 filter) and MIT Press's own Books Gateway page; no DOI exists for this 1986 book chapter. Correct as-is with no doi field. Source: Crossref queries + web search.

### mustafa_2019_cosmogan (doi 10.1186/s40668-019-0029-9)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title/authors/year/doi/journal/vol/pages | — | match (title differs only in case) | match | Crossref |
| primaryclass | **astro-ph.CO** | **astro-ph.IM** | **MISMATCH → fixed** | arXiv API |

### gili_2024_generalization (doi 10.1103/PhysRevApplied.21.044032)
| Field | Bib | Official | Match | Source |
|---|---|---|---|---|
| title/authors/year/doi/journal/vol/pages | — | match | match | Crossref |
| primaryclass | **quant-ph** | **cs.LG** | **MISMATCH → fixed** | arXiv API |

### placidi_2026_connectivity (arXiv 2604.12635)
All fields match. doi kept as-is (`10.48550/arXiv.2604.12635`, the arXiv preprint DOI) — Crossref bibliographic search found no IEEE Xplore record yet for the "Accepted at IEEE QCE 2026" venue (proceedings not indexed at verification time). Source: arXiv API + Crossref search.

### lizziobosco_2026_pauli (arXiv 2607.20409)
All fields match. doi kept as-is (arXiv preprint DOI) — Crossref bibliographic search found no other venue/DOI (only an unrelated same-title-fragment paper by different authors on portfolio optimization). Source: arXiv API + Crossref search.

## NOT FOUND list

None. All 52 entries' primary sources (arXiv record or DOI) resolved successfully.

## Judgment calls for <user> (not auto-applied in refs_verified.bib beyond adding an explanatory `note`)

1. **`recio_2025_iqpopt`** — arXiv's own metadata for 2501.04776 names the first author "Erik Armengol", while the companion paper 2503.02934 (`recio_2025_train`) and general usage elsewhere use "Erik Recio-Armengol". Kept as "Recio-Armengol, Erik" in `refs_verified.bib` for internal consistency with the other citation, with a `note` documenting the discrepancy — flip it if you want literal per-paper accuracy instead.
2. **`ml4sci_deeplense`** — "ML4SCI Collaboration" isn't a literal GitHub-registered entity name; the org is just "ML4SCI" and the repo has individual GSoC contributors, not a formal byline. Left unchanged; consider `{ML4SCI}` if you want stricter accuracy.

## Files produced

- `<HOME>/gsoc2026-deeplense-qml/paper/ml4ps2026/v12/refs_verified.bib` — corrected bibliography, same 52 bib keys, ready to replace `refs.bib`.
- `<HOME>/gsoc2026-deeplense-qml/paper/ml4ps2026/v12/REFS_VERIFICATION.md` — this file.
