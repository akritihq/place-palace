# place-palace

Reproduction code for the PLACE / PALACE paper pair on certified
classification of persistence diagrams.

**Version 2.0.0-rc1 (2026-11-13).** The code now implements the paper's
definitions exactly (radius cap, bottleneck-metric FPS, one separation
scale tau, admissibility), and every PALACE table was re-run from it.
See [`CHANGELOG.md`](CHANGELOG.md) for each correction, its paper
reference and its measured consequence. Numbers published before
2026-11-08 reproduce only with the v1 code (commit `922c9d7`).

## Papers

- **PLACE** — *A Closed-Form Persistence-Landmark Pipeline for
  Certified Point-Cloud and Graph Classification.* Majhi, Mitra,
  Virk, Bagchi. arXiv:[2605.02836](https://arxiv.org/abs/2605.02836).
  Uniform-grid landmark placement; pairwise certificate
  $\lambda(\nu)\,d_\mathcal{B}(A,B)$.
- **PALACE** — *PALACE: Persistence-Adaptive Landmark Cover
  Embedding.* Majhi, Mitra, Virk, Bagchi.
  arXiv:[2605.04046](https://arxiv.org/abs/2605.04046). Class-aware FPS placement with the diagonal-clearance
  cap $r_k \le d_\mathcal{B}(p_k,\Delta)$; per-prediction agreement
  diagnostic $r_m < \tfrac14\widehat\Delta$; closed-form
  kernel-Mahalanobis filtration selector.

## Quick start

```bash
pip install -r requirements.txt
python experiments/test_paper2_definitions.py      # code vs. definitions, T1-T9 (~3 min, see below)
python experiments/exp_reproduce_orbit5k_90.py     # PALACE 90.4% baseline  (90.44 +/- 0.87)
python experiments/exp_orbit5k_push92.py           # PALACE Orbit5k sweep; headline cell 90.9 +/- 1.2
python experiments/full_selector_hierarchy.py      # selector validation across 10 datasets
```

### Definition test suite

`experiments/test_paper2_definitions.py` checks the implemented objects
against the paper's definitions and theorems, and prints the measured
size of any gap:

| Test | Checks |
|------|--------|
| T1 | `embed` $= 2^{-3/2}\,w_k\,\varphi_k$ and `rho_nu` $= 2^{-3/2}\,\tau\,w_{\min}/4$ (the implementation's normalisation, documented in `embedding/nonuniform.py`) |
| T2 | `utils.bottleneck.d_B1_batch` $=$ `eq:db_single` (brute force) |
| T3 | Lemma lip_coord: $\lvert\varphi(x)-\varphi(y)\rvert \le d_\mathcal{B}(x,y)$ |
| T4 | the cap $r_k \le d_\mathcal{B}(p_k,\Delta)$ of `eq:config_def` on the paper's run configurations (and the legacy violation rate with `enforce_cap=False`) |
| T5 | `eq:upper_distortion` holds under the cap; constructive failure without it |
| T6 | `farthest_point_sampling` runs in the metric of `eq:db_single` (`eq:fps_rule`) |
| T7 | `lebesgue_number` $=$ `eq:lebesgue`; `is_admissible` $=$ `def:sep_radius` (i)+(ii) |
| T8 | Theorem nu_rho on real cross-class pairs with exact (gudhi) $d_\mathcal{B}$; the audit script's qualification against it |
| T9 | the three certificate radii in `grid_common` / `recompute_cert_firing_quarter` equal the paper's formulas |

Run it from the repository root:

```bash
python experiments/test_paper2_definitions.py
```

It needs the MUTAG degree + HKS diagram caches
(`data/diagrams/MUTAG_{degree,hks_t10}_H0H1_ext.pkl`, built by
`experiments/exp_noninterference_audit.py` from the TU download) and one
Orbit5k top-50 cache (`data/cache/orbit5k_alpha_H1_top50.pkl`, built by
`experiments/exp_orbit5k_push92.py` or `experiments/exp_orbit5k_best.py`);
it exits 1 on any FAIL. The
log of the run behind the current version is
`results/test_paper2_definitions_2026-11-08.txt` (ALL PASS).

## Reproduction map (PALACE, current version)

Scripts run from the repository root and write under `results/`. The
result files listed are the ones behind the tables (per-fold where the
table reports a mean); they are tracked in this repository.

| Table / figure | Script | Results file(s) |
|----------------|--------|-----------------|
| Tab. `push92` (Orbit5k sweep, 3 seeds x 10 folds; headline cell 90.9 +/- 1.2, sweep max 91.2 +/- 1.3) | `experiments/exp_orbit5k_push92.py` (`PUSH92_SEEDS=42,43,44`; per-configuration runs merged by `experiments/push92_merge.py`) | `results/orbit5k_final_sweep/push92_rerun.csv`, `push92_folds_rerun.csv` |
| Tab. `palace_headline` (90.44 +/- 0.87 baseline) | `experiments/exp_reproduce_orbit5k_90.py` | `results/orbit5k_reproduce_90_rerun.txt` |
| Tab. `graph_comparison` (all ten sets, FPS K = 200, alpha = 0.75, cap; rho-Mah and nested-CV selectors; 5 seeds x 10 folds) | `experiments/analyze_gamma_filt.py` per (dataset, seed, fold) -> `experiments/merge_seed_files.py` -> `experiments/exp_outer_test_selected_filt.py` | `results/filtration_grid/analysis/outer_test_selected_<ds>.csv` (per fold), `gamma_filt_<ds>_with_mah_seed*_part*.csv` (selector statistics per fold), `gamma_filt_<ds>[_full46]_with_mah.csv` (merged), `results/rerun_outer_test_<DS>.txt` |
| Tab. `graph_comparison` (PI-GIN column) | `experiments/exp_tda_gnn_baseline_pii.py`, `embedding/pi_gin.py` | `results/paper_II/tables/tda_gnn_baseline.csv` |
| Tab. `certificate_firing` (valid vs plug-in radii on the FPS embedding, sample-split; 0/11) | `experiments/cert_firing_valid_fps.py --dataset <DS>` | `results/paper_II/cert_firing_valid_fps_<DS>.csv` (one row per seed x fold), `.txt` (summary) |
| Preprint's plug-in firing table (uniform-grid embedding, "3 of 11"); kept for the comparison in Sec. 5 | `experiments/recompute_cert_firing_quarter.py` (needs the v1 grid records `results/filtration_grid/*_records_newweights.csv` and `analysis/full46_newweights/mahalanobis_*.csv`, not shipped; produced by `experiments/exp_grid_*.py`) | — |
| Tab. `nu_coherence_pii` (exact $d_\mathcal{B}$, one tau) | `experiments/audit_nu_coherence_pii.py` | `results/paper_II/tables/tab_nu_coherence_pii.csv` (v1: `tab_nu_coherence_pii_legacy.csv`) |
| Tab. `cor32_regularity` | `experiments/audit_cor32_regularity.py` | `results/paper_II/tables/tab_cor32_regularity.csv`, `results/rerun_audit_cor32_regularity.txt` |
| Tab. `fps_radius_ablation` (alpha immaterial under the cap) | `experiments/exp_fps_radius_ablation_pii.py` | `results/paper_II/tables/fps_radius_ablation.csv`, `results/rerun_exp_fps_radius_ablation_pii.txt` |
| Tab. `certificate_bound_audit` | `experiments/exp_certificate_bound_audit.py` | `results/paper_II/certificate_bound_audit.csv`, `results/rerun_exp_certificate_bound_audit.txt` |
| Non-interference audit | `experiments/exp_noninterference_audit.py` | `results/paper_II/noninterference_audit.csv`, `results/rerun_exp_noninterference_audit.txt` |
| Tab. `domain_inflation` | `experiments/exp_domain_inflation.py` | `results/rerun_exp_domain_inflation.txt` |
| Tab. `h0_joint_feature` | `experiments/exp_pv_joint_feature.py` | `results/rerun_exp_pv_joint_feature.txt` |
| Code-vs-definitions tests | `experiments/test_paper2_definitions.py` | `results/test_paper2_definitions_2026-11-08.txt` |
| Tab. `selection_statistics` (selector ranks; pp vs pooled) | `experiments/full_selector_hierarchy.py`, `experiments/compare_fisher_pp_vs_pool.py` | `results/paper_II/tables/tab_selection_statistics.tex` (v1 run, see below) |
| Orbit5k headline grid | `experiments/exp_orbit5k_best.py` | — |
| $\hat\gamma$ kernel-margin sanity | `experiments/exp_palace_gamma_sanity.py` | — |
| PALACE vs landscape MMD power | `experiments/exp_palace_vs_landscape_power.py` | — |
| Sec. 6.2 small-$K$ MUTAG sweep | `experiments/exp_mutag_smallK.py` | — |

Seven secondary sweeps (`orbit_sweep`, `continuation`, `alpha_gamma`,
`filtration_gamma`, `selection_statistics`, `complementarity`, `ksigma`)
were not re-run after the cap; the paper says so in a footnote. Their
shipped outputs (`results/paper_II/tables/tab_selection_statistics.tex`,
`tab_complementarity.tex`, `tab_ksigma_sensitivity.tex`) are v1 runs and
do not reproduce with the v2 code (see `CHANGELOG.md`, "Reproducing v1
numbers"). Everything else in the map is from the v2 code.

### Cluster drivers

`experiments/cluster/` holds the SLURM array scripts of the re-run
(`wave_q.sbatch`, `wave_q3.sbatch`, `wave_q2b_outer.sbatch`,
`wave_r.sbatch`; account and mail address redacted). They assume the
private repository's layout (`cluster/` at the root, `PLEB_HOME`, a
conda environment `pleb-env`); adapt the paths to run them here. The
header comments give the exact task-to-script mapping and submit lines.

### Paper I (PLACE) rows, unchanged

| Table / Figure | Script |
|----------------|--------|
| Paper I Tab. `certificate_bound_audit` | `experiments/exp_pi_certificate_bound_audit.py` |
| Paper I Tab. `coherence_audit` | `experiments/exp_pi_coherence_audit.py` |
| Paper I Tab. `cert_firing` | `experiments/regen_cert_firing_table_honest.py` |
| Paper I honest per-fold firing (Rem. 5.3) | `experiments/recompute_cert_firing_honest_perfold.py` |
| Paper I Rem. 5.2 stable-rank audit | `experiments/audit_stable_rank_HW.py` |
| Paper I Tab. `exp1` closed-form Mah selector | `experiments/selector_ablation_committed.py` |
| Paper I Sec. 6 Ballester--Rieck VR demo | `experiments/exp_ballester_rieck_demo.py` |
| Paper I Figs. `intro_pd` / `orbits` | `figures/mutag_graph_to_diagram.pdf`, `figures/orbit5k_examples.pdf` |
| Graph grids (v1 uniform-grid lower block; superseded by the FPS outer test above) | `experiments/exp_grid_{proteins,imdbb,imdbm,nci109}.py`, `experiments/cluster_graph_classifiers.py`, `experiments/aggregate_certificate_firing.py` |

## Data

Datasets download on first use (TU Dortmund sets into `data/raw/`,
processed into `data/processed/`; Orbit5k is generated). Diagram caches
are written under `data/` and are not tracked. Intermediate embedding
caches land under `results/` next to the tracked result files.

## Citation

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

MIT — see [`LICENSE`](LICENSE).
