# Objective 3 (OPTIONAL) — equity-constrained resource allocation

P4's track. working.md §3: allocate limited relief across the 16 study wards so that
wards with high social vulnerability aren't starved just because they generate less
social-media distress signal.

## `fetch_census.py` + `build_vulnerability_index.py` — task 7.1

Builds a ward-level **social vulnerability index** from real Census 2011 proxies
(working.md §3.2: "population density, literacy, sanitation access"). This is
*vulnerability* only (who copes worst), kept separate from flood *exposure*
(Objective 1's graph state); task 7.2 combines the two.

```bash
python src/optimizer/fetch_census.py              # downloads + extracts Census inputs (needs openpyxl, xlrd)
python src/optimizer/build_vulnerability_index.py
```

### The data problem found first: the study wards don't exist in Census 2011

The 16 study wards use GCC's **200-ward** numbering, created by the October 2011
expansion. Census 2011 (Feb–Mar 2011) was enumerated on the **old 155-ward** Chennai
corporation. This was confirmed from the District Census Handbook (Part B) appendix,
which lists exactly `WARD No.-0001`…`0155`. You can't look a study ward up by number.

- **14 wards**: crosswalked spatially from the old wards using DataMeet's digitised
  pre-2011 boundaries (`Wards-2008.geojson`, CC BY-SA 2.5 IN). 8 of the 14 are ≥90% one
  dominant old ward. The rest are genuine blends: 142 (60%), 172 (71%), 175 (76%),
  173 (80%), 171 (84%), 176 (85%).
- **Wards 168 and 169**: outside the 2011 corporation entirely. Despite the "PERUNGUDI"
  zone label, the gazetteer's own localities place them in **Ullagaram (168)** and
  **Puzhuthivakkam (169)**. In 2011 that was the separate *Puzhithivakkam (Ullagaram)
  Municipality*, Kancheepuram district, Census town 803357. It is **not** Perungudi TP
  (803358), which the zone name would suggest. Public records confirm it was annexed as
  exactly wards 168 + 169.

Counts, never rates, are crosswalked by **areal weighting**: each source unit's counts
are split in proportion to the share of *its own* area that each new ward covers. Rates
are computed only after reaggregation, so a ward spanning several old wards gets a
population-weighted rate, not an average of averages. Study-area total: **691,321
people**. The source files sum to 4,646,732 (Chennai M Corp.) + 53,322 (Puzhithivakkam),
which matches the published district total exactly.

**Real resolution limit, measured:** the 16 wards rest on only **12 distinct dominant
census units**. 177/178 (old ward 153), 181/182 (old ward 155) and 168/169 (the
municipality) share near-identical rates by construction.

### Indicators (higher = more vulnerable)

| Indicator | Census source | Why |
|---|---|---|
| Population density | PCA `TOT_P` / km² | people exposed per unit area |
| Illiteracy, age 7+ | `1 − P_LIT / (TOT_P − P_06)` | Census's own literacy definition; access to warnings |
| SC/ST share | `(P_SC + P_ST) / TOT_P` | social marginalisation |
| Children 0–6 | `P_06 / TOT_P` | dependants; evacuation burden |
| Marginal workers | `MARGWORK_P / TOT_WORK_P` | precarious (<6 months/yr) work |
| Household size | `TOT_P / No_HH` | crowding |
| No piped-sewer latrine | HH-08, **town level only** | septic tanks overflow in floods |

**Sanitation exists only at town level.** HH-08 (latrine facility) is published down to
towns, not wards. So it can only separate 168/169 (Puzhithivakkam: **89.7%** of households
without a piped-sewer latrine) from the other 14 (Chennai M Corp.: **7.3%**). It's kept
because the gap is large and real, but the index is also reported without it. The other
HH-08 column, "no latrine in premises", was rejected because it points the *opposite* way
(Puzhithivakkam 2.0% vs Chennai 4.4%) and measures in-house convenience, not what happens
to sewage when a ward floods.

**Method:** min-max normalise each indicator across the 16 wards, then take the
unweighted mean (SoVI-style default, no invented weights).

### Result

| Rank | Ward | Locality | Index | Driven by |
|---|---|---|---|---|
| 1 | 180 | Taramani | 0.58 | highest illiteracy (12%) and marginal workers (16%) |
| 2 | 142 | Saidapet | 0.50 | density 45,424/km², 2.2× the next-densest ward |
| 3 | 171 | Little Mount / Jothiammal Nagar | 0.48 | highest SC/ST share (24%) |
| 4 | 170 | Ekkattuthangal / Labour Colony | 0.45 | illiteracy 11%, SC/ST 17% |
| … | | | | |
| 14–16 | 182, 181, 176 | Thiruvanmiyur, Besant Nagar / Adyar | ~0.20 | low on nearly everything |

Figure: `docs/figures/phase7_vulnerability_index.png` (map + per-indicator heatmap).
Face validity, stated as qualitative and not as proof: the top wards hold riverbank and
canal-side settlements, and the bottom wards are well-known affluent localities.

### Robustness: does the ranking survive changing the judgment calls?

There are 10 rankings in total: the primary one, a PCA-weighted variant, 7
leave-one-indicator-out variants, and a road-length dasymetric crosswalk.

- Spearman ρ vs. primary: **min 0.90, median 0.93**. No variant reshuffles the ranking.
- **Robust core**: wards **180, 142, 171, 170** are in the top 5 in ≥80% of rankings, and
  180 is in all 10. Slot 5 is a genuine near-tie (179: 0.405, 177/178: 0.404, 172: 0.385),
  so treat it as soft.
- **PCA weighting**: PC1 explains 35% and loads on illiteracy (0.25) + SC/ST share (0.24),
  which correlate at ρ=0.75. Equal weights therefore partly double-count one underlying
  factor. The PCA-weighted ranking still agrees at ρ=0.90.
- **Dasymetric crosswalk**: for the 11 source units ≥95% inside the study area, people are
  split by road length (the project's own 17,195 segments) instead of area. The ranking
  barely moves (ρ=0.985), but **population counts do**: ward 174 drops 61k → 46k (its
  8.3 km² includes large low-density institutional land) and 142 rises 52k → 61k. Both
  estimates are in the CSV (`population`, `population_dasymetric`); task 7.2 should
  carry that uncertainty, not pick one silently.
- **External validation: not possible at ward level.** Census 2011's slum PCA (NADA 6190)
  and amenity tables are town-level only. The stability analysis tests whether the ranking
  is an artifact of method choices; it can't test whether the proxies themselves are right.

### Outputs

- `data/raw/census/` (committed): `pca_wards_2011.csv`, `sanitation_towns_2011.csv`,
  `chennai_wards_2008.geojson`, `census_sources.json` (URL + sha256 of every download).
  The full district source files under `source/` are gitignored; rerun the fetch script.
- `data/processed/optimizer/` (gitignored, regenerate): `ward_vulnerability_index.csv` /
  `.geojson` (all indicators, index variants, crosswalk diagnostics, top-5 frequency) and
  `vulnerability_index_report.json`.

**Network note:** `curl` on this machine needs `--ssl-revoke-best-effort` for
censusindia.gov.in (the Sophos proxy breaks Windows' revocation check). Python
`requests` works as-is.

## `equity_optimizer.py` — task 7.2

The constrained optimizer itself (OR-Tools MIP, SCIP backend). It combines Objective 1's
flood state, Objective 2's distress signals and 7.1's vulnerability index into one ward
allocation.

```bash
python src/optimizer/equity_optimizer.py
```

### The bias, measured in this project's own data

Distress signals per ward come from Objective 2's real pipeline output: **10 real distress
posts**, mapped to wards. Reported distress clusters where news coverage of 2015 went:
Velachery (179: 6 posts) and Kotturpuram (172: 4). **3 of 7.1's 4 robust-core
vulnerable wards (180 Taramani, 171 Little Mount, 170 Ekkattuthangal) produce zero
signal.** The zero-signal wards together hold **45.6% of all affected people**.

To be precise about the claim: across all 16 wards, Spearman(vulnerability, signals) is
+0.18. Signal isn't systematically *anti*-correlated with vulnerability. The failure is
concentrated in the most vulnerable wards, which is exactly where it matters.

Two input filters were found while building this, not assumed:
- **Smoke-test rows:** task 4.5's pipeline output also contains its **4 synthetic
  smoke-test sentences**, one a located "Families in Velachery were trapped..." distress
  post. Only rows whose text is in task 1.13's real corpus count.
- **Known false positive:** task 5.5's Nandambakkam → Adambakkam match (ward 177) is dropped.

(The dashboard's Distress Signals tab currently has neither filter. That's a separate fix,
not done here.)

### Model

- **Need:** need_i = population_i (7.1) × ward i's road-length share flooded at **peak** in
  the fused ground truth. Total: 628,140 affected people. Ground truth is the primary
  exposure because relief is allocated during or after the peak, when the observed extent
  exists. The GNN forecast is a sensitivity variant, since its per-ward probabilities
  barely discriminate (tasks 5.1/5.3).
- **Solver:** integer relief units x_i, budget B = 25% of need (157,035 units).
  Maximise Σ p_i x_i, where p_i = share of reported distress.
- **Constraints:** Σx ≤ B, 0 ≤ x_i ≤ need_i, and the **equity floor**
  x_i ≥ ρ · c̄ · (v_i / v_max) · need_i, where c̄ = B / Σneed is the even-split coverage.
- **What ρ means:** ρ = 1 guarantees the most vulnerable ward at least average coverage.
  The floors are feasible for every ρ ∈ [0, 1] by construction. ρ = 0 is the same model
  with the constraint off, and above the floors the budget still follows reported demand.

### Result (budget = 25% of need)

| Model | Starved wards (<5% coverage) | Core min coverage | Zero-signal wards' share of relief (vs. 45.6% of need) | Reported-distress coverage | Coverage Gini |
|---|---|---|---|---|---|
| Demand-driven optimum (ρ=0) | **12 / 16** | 0% | **0%** | 72.9% | 0.74 |
| Signal-proportional (dispatcher heuristic) | **8 / 16** | 0% | **0%** | 71.7% | 0.65 |
| Need-proportional (reference) | 0 | 25.0% | 45.6% | 25.0% | 0.00 |
| **Equity-constrained (ρ=1)** | **0** | **19.4%** | **29.8%** | 55.5% | 0.40 |

The signal-proportional heuristic is included so the comparison isn't against a
strawman: a dispatcher splitting by call volume starves the same three core wards as
the optimiser does.

**Price-of-fairness frontier (ρ = 0 → 1):** ρ = 0.6 is the cheapest setting with **zero
starved wards**, at only **7.2%** of reported-distress coverage given up. ρ = 1 costs
23.9%. The knee is marked on the frontier plot (`cheapest_no_starvation_rho` in the
report). That's a concrete policy choice, not a single number.

**An aggregate fairness metric would have hidden the problem.** The demand-driven optimum
scores *higher* on aggregate vulnerability-weighted coverage (0.278 vs 0.269) because it
fully covers the moderately vulnerable, high-signal wards 142/172/179, while giving three
core wards nothing. That's why the model enforces per-ward floors rather than optimising
an aggregate fairness score.

### Sensitivity: the headline holds in every scenario

The scenarios are 3 budgets (10/25/50%), GNN-forecast exposure, the dasymetric population
estimate, and the index without sanitation or PCA-weighted. In every one, demand-driven
starves **3–4** robust-core wards and the equity model starves **0**. Price of fairness
(ρ=1) ranges 14–49%; it is highest at the tightest budget, where every unit given to a
floor is a unit taken from a loud ward.

At a 10% budget the equity model leaves 4 *non-core* wards under the 5% "starved" line.
This is by design: c̄ = 10%, so the least-vulnerable wards' floors fall below 5%. The core
stays protected.

### Outputs

- **Figure:** `docs/figures/phase7_allocation_comparison.png`, with per-ward coverage by
  model on the left and the frontier with the knee marked on the right.
- **`data/processed/optimizer/` (gitignored):**
  - `ward_allocation_inputs.csv`: need, signals, vulnerability, floors, and GNN-forecast
    flooded share.
  - `allocation_comparison.csv`: units and coverage per ward for each model.
  - `equity_optimizer_report.json`: signal audit, models, frontier and sensitivity.

**Why OR-Tools, not PuLP:** the installed PuLP is 4.x, a Rust-core rewrite whose
`LpVariable` API is incompatible with the documented 2.x one and ships no CBC binary.
OR-Tools (`pywraplp`, SCIP) was the other solver working.md §3.4 names, and it's already
in `requirements.txt`.
