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
| Paper I certificate audit                             | `experiments/exp_pi_certificate_bound_audit.py` |

Intermediate caches are regenerated on first run; outputs are
written under `results/` (gitignored).

## Citation

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

MIT — see [`LICENSE`](LICENSE).
