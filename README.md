# place-palace

Reproduction code for the PLACE / PALACE paper pair on certified
classification of persistence diagrams.

## Papers

- **PLACE** — *A Closed-Form Persistence-Landmark Pipeline for
  Certified Point-Cloud and Graph Classification.* Majhi, Mitra,
  Virk, Bagchi. arXiv:[2605.02836](https://arxiv.org/abs/2605.02836).
  Uniform-grid landmark placement; pairwise certificate
  $\lambda(\nu)\,d_\mathcal{B}(A,B)$.
- **PALACE** — *PALACE: Persistence-Adaptive Landmark Cover
  Embedding.* Majhi, Mitra, Virk, Bagchi.
  arXiv:[2605.04046](https://arxiv.org/abs/2605.04046).
  Class-aware FPS placement; per-prediction certificate
  $r_m < \tfrac{1}{2}\widehat\Delta_{\hat c}$; closed-form
  kernel-Mahalanobis filtration selector.

## Quick start

```bash
pip install -r requirements.txt
python experiments/exp_reproduce_orbit5k_90.py     # PALACE 90.4% baseline
python experiments/exp_orbit5k_push92.py           # PALACE 91.3% certified
python experiments/full_selector_hierarchy.py      # selector validation: rho_Mah +0.66 across 10 datasets
```

## Reproduction map

| Table / Figure                                        | Script                                          |
|-------------------------------------------------------|-------------------------------------------------|
| Paper II Tab. `palace_headline`                       | `experiments/exp_reproduce_orbit5k_90.py`       |
| Paper II Tab. `push92`                                | `experiments/exp_orbit5k_push92.py`             |
| Paper II Tab. `domain_inflation`                      | `experiments/exp_domain_inflation.py`           |
| Paper II Tab. `certificate_bound_audit`               | `experiments/exp_certificate_bound_audit.py`    |
| Paper II Tab. `certificate_firing`                    | `experiments/aggregate_certificate_firing.py`   |
| Paper II Tab. `selection_statistics` (selector ranks) | `experiments/full_selector_hierarchy.py`        |
| Paper II Tab. `selection_statistics` (pp vs pooled)   | `experiments/compare_fisher_pp_vs_pool.py`      |
| Paper II Orbit5k headline grid                        | `experiments/exp_orbit5k_best.py`               |
| Paper II $\hat\gamma$ kernel-margin sanity            | `experiments/exp_palace_gamma_sanity.py`        |
| Paper II PALACE vs landscape MMD power                | `experiments/exp_palace_vs_landscape_power.py`  |
| Paper II §6.2 small-$K$ MUTAG sweep                   | `experiments/exp_mutag_smallK.py`               |
| Paper II non-interference audit                       | `experiments/exp_noninterference_audit.py`      |
| Paper II Tab. `nu_coherence_pii`                      | `experiments/audit_nu_coherence_pii.py`         |
| Paper II Tab. `cor32_regularity`                      | `experiments/audit_cor32_regularity.py`         |
| Paper II Tab. `fps_radius_ablation`                   | `experiments/exp_fps_radius_ablation_pii.py`    |
| Paper II Tab. `h0_joint_feature`                      | `experiments/exp_pv_joint_feature.py`           |
| Paper II Tab. `graph_comparison` (PI-GIN)             | `experiments/exp_tda_gnn_baseline_pii.py`, `embedding/pi_gin.py` |
| Paper II graph rows (PROTEINS/DD/IMDB/NCI109)         | `experiments/exp_grid_{proteins,imdbb,imdbm,nci109}.py`, `experiments/cluster_graph_classifiers.py` |
| Paper I Tab. `certificate_bound_audit`                | `experiments/exp_pi_certificate_bound_audit.py` |
| Paper I Tab. `coherence_audit`                        | `experiments/exp_pi_coherence_audit.py`         |
| Paper I Tab. `cert_firing`                            | `experiments/regen_cert_firing_table_honest.py` |
| Paper I honest per-fold firing (Rem. 5.3)             | `experiments/recompute_cert_firing_honest_perfold.py` |
| Paper I Rem. 5.2 stable-rank audit                    | `experiments/audit_stable_rank_HW.py`           |
| Paper I Tab. `exp1` closed-form Mah selector          | `experiments/selector_ablation_committed.py`    |
| Paper I §6 Ballester--Rieck VR demo                   | `experiments/exp_ballester_rieck_demo.py`       |
| Paper I Figs. `intro_pd` / `orbits`                   | `figures/mutag_graph_to_diagram.pdf`, `figures/orbit5k_examples.pdf` |

Intermediate caches and raw fold-level accuracies are written under
`results/` (gitignored); they regenerate on first run of the scripts
above. The anonymized supplementary attached to each submission
additionally ships the exact fold-level CSVs.

## Citation

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

MIT — see [`LICENSE`](LICENSE).
