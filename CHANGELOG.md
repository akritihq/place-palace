# Changelog

## v2.0.0-rc1 (2026-11-13)

The code is made to implement the definitions of the PALACE paper (current version),
and every number the paper reports is re-derived from it. A review of 2026-11-08
executed the released code against the paper's definitions (something the earlier
paper-vs-CSV checks had not done) and found the mismatches below; none of them is
visible in classification accuracy, all of them affect the stated guarantees or the
reported numbers. The numbering follows that review. Paper references (`eq:`, `def:`,
`thm:`) are the labels of the current manuscript.

### Corrections to `embedding/nonuniform.py`

1. **The radius cap of `eq:config_def` is now enforced** (finding 1).
   `init_nonuniform_from_data` and `init_nonuniform_class_weighted` apply
   r_k <= d_B(p_k, Delta) after the [tau/2, 4 tau] clip (`enforce_cap=True`, the default).
   Before, the cap was never applied: at K = 200 it was violated by 53.5% of the landmarks
   on MUTAG, 42% on PTC, 21.5% on COX2, 20-21.5% on DHFR, and 18% (alpha filtration) /
   30% (density-k10) on Orbit5k. On the chemical sets every violator sat at the lower clip
   tau/2. Radii exceeded the landmark's diagonal distance by up to 18x, and the upper
   distortion bound `eq:upper_distortion` (whose proof needs phi(Delta) = 0, i.e. the cap)
   failed by 48-165x on a constructed pair. Under the cap the bound holds with a 30x margin
   (test T5a). Measured cost: about half an accuracy point on Orbit5k (Table push92).
   `enforce_cap=False` is kept for the record (test T5b) but does not reproduce v1, see
   item 2.
2. **FPS runs in the bottleneck metric of `eq:db_single`** (finding 6).
   `farthest_point_sampling` and `soft_grad_positions` (plus its gradient branch) use the
   via-diagonal cost max(pers_p, pers_q); v1 used the sum. With the sum, the K = 200
   landmark sets differed from the max-rule greedy that `thm:fps_greedy` is about on
   82/200 (MUTAG), 55 (PTC), 71 (COX2), 103 (DHFR) and 128 (Orbit5k) selections, with a
   2-8% worse true d_B covering radius. **Landmark selections therefore differ from v1
   for every configuration**, and all tables were re-run.
3. **One tau** (finding 3). tau is the median half-persistence of the training points
   (the clip scale of the paper's protocol table). It is stored on the embedding as
   `emb.tau`, together with `emb.alpha`, `emb.cap_enforced` and the training support, and
   every downstream script reuses it. v1 used three different quantities; in particular the
   nu-coherence audit used the median clipped radius, which equals tau/2 exactly on the
   chemical sets (95-100% of radii sat on the lower clip), so its tau, rho_nu = tau/(4 sqrt K)
   and the admissibility verdict differed from the protocol's by 2x.
4. **`admissibility()` / `is_admissible()` implement `def:sep_radius`** (finding 7):
   (i) lambda_0 >= tau/4 and (ii) r_k <= (tau + lambda_0)/2, on the stored training support,
   via `lebesgue_number` (which equals `eq:lebesgue` to machine precision, test T7a). v1's
   `is_admissible` checked min r >= gamma, max r <= 4 tau and rho_nu > gamma, which is not
   the definition and disagreed with it on 4 of 9 (configuration, tau) cases; it was never
   called by an experiment.
5. **The 2^{-n-1/2} normalisation is documented** (finding 5). `embed` returns
   w_k 2^{-n-1/2} phi_{p_k, r_k} and `rho_nu` returns 2^{-n-5/2} tau w_min, a factor that the
   paper's `eq:sum_embed` / `eq:nu_rho_formula` do not carry. The factor is kept (sigma = 1e-3
   is fixed, so removing it would change the kernel, not only the scale) and stated in the
   module docstring; any comparison of a code-scale quantity with a paper-scale constant
   (Lipschitz constant N_max, floor w_k tau/4) must include it. This is where v1's audit floor (item 6) went wrong by 2.83x.

### Corrections to the experiment scripts

6. `audit_nu_coherence_pii.py` (finding 4). Pair qualification d_B(A, B) >= tau now uses the
   exact `gudhi.bottleneck_distance`; v1's `db_pair` had no bijection constraint and a
   sum-via-diagonal cost (median relative error -54% to -97% against the exact value; on
   MUTAG it accepted 4 pairs with d_B < tau and rejected 3016 of 3121 pairs that qualify).
   The floor |Phi_k(A) - Phi_k(B)| >= w_k tau/4 is tested on the paper's scale via the raw
   coordinates (v1 compared code-scale differences with the paper-scale floor: 2.83x too
   strict). The coherence rate stays 100% on every set; the examined/qualifying counts and
   tau change. The v1 table is kept as `tab_nu_coherence_pii_legacy.csv`.
7. Pinelis radius (i) (finding 10). `grid_common.compute_r_m` and
   `recompute_cert_firing_quarter.py` restore the centering factor 2 of the paper's
   2 R sqrt(2 log(2k/delta)/m_min); v1 had R sqrt(2 L/m). 0/11 firing either way.
8. `exp_orbit5k_push92.py` (finding 13). Several outer-CV seeds (`PUSH92_SEEDS`), per-fold
   accuracies written to `push92_folds*.csv`, sample standard deviation (ddof = 1). v1
   reported a population std over the ten folds of a single seed and stored no per-fold
   values. `push92_merge.py` merges per-configuration runs; `PUSH92_CONFIGS` selects cells so
   a sweep can run as short cluster tasks.
9. New `cert_firing_valid_fps.py` (findings 2, 3, 8, 9, 11, 12). The per-prediction
   certificate evaluated on the FPS K = 200 embedding the paper advertises (v1 evaluated it
   on the uniform-grid embedding), with the configuration fitted on a held-out half of each
   training fold so that the theorems' "configuration fixed independently of the sample"
   holds, the exact almost-sure R-bar, tr Sigma_c, a valid empirical-Bernstein covariance
   term, the fold-level condition r < Delta-hat/4, and per-fold records.
10. `analyze_gamma_filt.py` + `merge_seed_files.py` + `exp_outer_test_selected_filt.py`.
    The graph table is recomputed as outer-fold test accuracy at the selected filtration for
    all ten sets on the FPS embedding (v1's lower block PROTEINS/DD/IMDB-B/IMDB-M/NCI109 came
    from the uniform grid). DD attribution fixed (finding 8): v1's "DD fires at 100% of 50
    splits at the rho-Mah-selected filtration" was evaluated at a runner-up filtration on 42
    of the 50 folds because the selector's true pick had no certificate records; PROTEINS
    agreed with the true pick on 0/50 folds, IMDB-B 40%, IMDB-M 36%.
11. New `test_paper2_definitions.py`: tests T1-T9 of the code against the definitions and
    theorems (embedding and rho_nu scale, `d_B1_batch` = `eq:db_single`, Lemma lip_coord, the
    cap, `eq:upper_distortion` with and without the cap, the FPS metric, `eq:lebesgue` and
    admissibility, Theorem nu_rho on exact-d_B pairs, the certificate radii). Log in
    `results/test_paper2_definitions_2026-11-08.txt`: ALL PASS.

### Measured consequences (re-run 2026-11-09 to 2026-11-12; cap enforced, FPS in d_B, per-fold storage)

- **The certificate firing table is a plug-in diagnostic.** v1's "fires on 3 of 11 data sets"
  (NCI1 and DD at 100% of folds, NCI109 at 23%) was a property of the uniform-grid embedding
  with centroids and configuration from the same fold. On the FPS embedding with sample
  splitting (`cert_firing_valid_fps.py`, 5 seeds x 10 folds, 11 sets) the proof-valid radius
  certifies **0 of 11**, and so does the paper's own plug-in radius (iii): median
  radius/threshold is 13x-937x for the valid form, 2.5-2.8x for the trace plug-in on
  NCI1/DD/NCI109 (4.6-18x elsewhere) and 1.3-1.7x for its operator-norm variant on those
  three sets. The per-point valid certificate covers 0% everywhere. The plug-in correction is
  not of lower order: the prefactor R-bar/sqrt(tr Sigma_c) is 25-260 with the almost-sure
  R-bar (5.5-7.6 with the observed max norm).
- **Orbit5k headline.** The declared cell (triple_k15_top50, K = 200, alpha = 4.0) is
  **90.9 +/- 1.2** (3 seeds x 10 folds, sample s.d.), replacing 91.3 +/- 1.0 (one seed,
  population s.d.). The maximum over the 24-cell sweep is **91.2 +/- 1.3** (triple_k15_top100,
  K = 300, alpha = 4.0). Mean change over the sweep -0.12 pt; four cells >= 91.0 (were about
  ten). The 90.4 baseline reproduces as 90.44 +/- 0.87.
- **Graph table**, all ten sets on one method (PALACE FPS, K = 200, alpha = 0.75, cap),
  5 seeds x 10 folds, rho-Mah / nested-CV selector (v1 in parentheses): MUTAG 83.0 +/- 6.9 /
  86.4 +/- 7.8 (85.4 / 87.7); COX2 79.8 / 79.4 (81.0 / 79.1); DHFR 75.7 / 80.5 (75.1 / 80.2);
  PTC 60.2 / 57.0 (52.6 / 54.8); NCI1 71.4 / 70.8 (67.6 / 69.9); PROTEINS 71.4 / 70.6
  (grid 71.9); DD 75.8 / 75.9 (grid 75.7); IMDB-B 65.5 / 65.3 (grid 65.4); IMDB-M 43.2 / 43.5
  (grid 40.3); NCI109 71.3 / 70.3 (grid 67.1). The FPS embedding with the cap gains about
  4 points on NCI1/NCI109 and loses 2.4 on MUTAG; the rest is within fold noise.
- **alpha is immaterial under the cap** (`fps_radius_ablation`): COX2 78.2, DHFR 76.2, PTC 61.9
  at every alpha in [0.5, 3]; MUTAG 88.8 -> 88.5. With the cap most landmarks sit at
  r_k = d_B(p_k, Delta), so alpha only moves the few far-from-diagonal radii.
- **Audits qualitatively unchanged.** cor32 regularity: the worst-case hypothesis fails by
  10^3-10^4 (ratios rescaled to the proof's sigma^2 Delta^2 / 8 threshold), the variance-aware
  form holds on MUTAG and DHFR and fails on PTC and COX2. Non-interference strict form
  0-0.2%. Certificate-bound audit: rho_nu bound holds on 99.9-100% of qualifying pairs.
  Domain inflation identical to v1. nu-coherence 100% on all sets; Theorem nu_rho's minimum
  margin on MUTAG drops from 24x to 9x the floor. Measured stable rank per fold is 1.9
  (Orbit5k) to 7.6 (NCI109), not "<= 1.17".

### Known, left as is

- `CertifiedClassifier` in `embedding/nonuniform.py` (finding 14) is deprecated: it is not
  Algorithm 2, its radius is a plug-in diagnostic, and its former docstring claimed a
  correctness guarantee the theorems do not give. It now warns on construction; no paper
  script uses it.
- Seven secondary sweeps predate the cap and are footnoted as such in the paper
  (orbit_sweep, continuation, alpha_gamma, filtration_gamma, selection_statistics,
  complementarity, ksigma); the `.tex` tables under `results/paper_II/tables/` for
  complementarity, ksigma and selection_statistics are v1 outputs.
- `results/paper_II/tables/fps_radius_ablation.csv` holds 100 rows per (dataset, alpha) cell
  (the 5 seeds x 10 folds appear twice); the table means are over all 100.
- `recompute_cert_firing_quarter.py` (the v1 grid-embedding plug-in table) is shipped with the
  factor-2 fix but its inputs (`results/filtration_grid/*_records_newweights.csv`, the v1
  grid records) are not; see README.

### Reproducing v1 numbers

Every number reported before 2026-11-08 (arXiv v1 of PALACE, the preprint tables,
"91.3 +/- 1.0", "fires on 3 of 11", the v1 nu-coherence pair counts) was produced by the v1
code and reproduces **only** with it: check out the previous commit of this repository,
`922c9d7` (2026-07-12). The v2 code does not reproduce them even with `enforce_cap=False`,
because the FPS metric fix (item 2) changes every landmark selection and the certificate
scripts changed (items 7, 9, 10).
