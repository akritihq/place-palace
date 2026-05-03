# place-palace

Reproduction code for the PLACE / PALACE paper pair on certified
classification of persistence diagrams.

## Paper I (PLACE)

> *Persistence Landmarks: Certified Classification of Persistence Diagrams.*
> Bagchi, Majhi, Mitra, Virk.

PLACE places landmarks on a uniform grid over the persistence diagram
domain and lifts to an additive RKHS via a landmark kernel. The
certificate $\lambda(\nu)\,d_\mathcal{B}(A,B)$ holds on every
non-interfering pair.

### Reproduce the certificate audit

```bash
pip install -r requirements.txt
python experiments/exp_pi_certificate_bound_audit.py
```

Outputs are written to `results/`.
