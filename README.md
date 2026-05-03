# place-palace

Reproduction code for the PLACE / PALACE paper pair on certified
classification of persistence diagrams.

## Papers

- **PLACE** — *Persistence Landmarks: Certified Classification of
  Persistence Diagrams.* Bagchi, Majhi, Mitra, Virk.
  Uniform-grid landmark placement; pairwise certificate
  $\lambda(\nu)\,d_\mathcal{B}(A,B)$.
- **PALACE** — *Adaptive Landmarks for Certified Classification.*
  Class-aware FPS placement; per-prediction certificate
  $r_m < \tfrac{1}{2}\widehat\Delta_{\hat c}$.

## Quick start

```bash
pip install -r requirements.txt
python experiments/exp_reproduce_orbit5k_90.py     # PALACE 90.4% baseline
python experiments/exp_orbit5k_push92.py           # PALACE 91.3% certified
```

## Reproduction map

| Table / Figure                          | Script                                        |
|-----------------------------------------|-----------------------------------------------|
| Paper II Tab. `palace_headline`         | `experiments/exp_reproduce_orbit5k_90.py`     |
| Paper II Tab. `push92`                  | `experiments/exp_orbit5k_push92.py`           |
| Paper II Tab. `domain_inflation`        | `experiments/exp_domain_inflation.py`         |
| Paper II Tab. `certificate_bound_audit` | `experiments/exp_certificate_bound_audit.py`  |
| Paper II Tab. `certificate_firing`      | `experiments/aggregate_certificate_firing.py` |
| Paper II §6.2 small-K MUTAG sweep       | `experiments/exp_mutag_smallK.py`             |
| Paper II non-interference audit         | `experiments/exp_noninterference_audit.py`    |
| Paper I certificate audit               | `experiments/exp_pi_certificate_bound_audit.py` |

Intermediate caches are regenerated on first run; outputs are
written under `results/` (gitignored).

## Citation

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

MIT — see [`LICENSE`](LICENSE).
