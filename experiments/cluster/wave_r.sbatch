#!/bin/bash
#SBATCH --job-name=p2-rerun
#SBATCH --account=REDACTED_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=cluster/logs/waver_%A_%a.out
#SBATCH --error=cluster/logs/waver_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=REDACTED
#
# Paper II re-run, fine-grained so every task backfills (2026-11-09: 12-24 h requests sat >13 h while
# 1.5 h / 32 GB probes started at once).  Same science as wave_q.sbatch.  PLEB_HOME=$HOME/pleb2.
#
#   0-71     push92: seed = 42 + i//24, config = i%24                       10 cpu 32G 1:30   [Q1]
#   72       exp_reproduce_orbit5k_90                                         10 cpu 32G 2:00   [Q1]
#   73-322   analyze_gamma_filt chem: ds=(i-73)//50, seed=((i-73)%50)//10, fold=(i-73)%10   8 cpu 32G 1:30 [Q2]
#   323-572  analyze_gamma_filt social: same layout from 323                 8 cpu 48G 2:00   [Q2b]
#   573-578  audits (cor32, fps_radius_ablation, noninterference, certificate_bound, domain_inflation, pv_joint)  8 cpu 32G 2:00 (ablation 4:00)
#   579-589  cert_firing_valid_fps, 11 sets                                  8 cpu 32G 1:30   [Q3]
#   590-594  merge + outer test, chem (afterok 73-322)                       8 cpu 32G 2:00   [Q2]
#   595-599  merge + outer test, social (afterok 323-572)                    8 cpu 48G 3:00   [Q2b]
#
set -euo pipefail
module purge; module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/pleb-env/bin/python"; export PATH="$HOME/.conda/envs/pleb-env/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${PLEB_HOME:-$HOME/pleb}"
mkdir -p cluster/logs results/orbit5k_final_sweep results/filtration_grid/analysis results/paper_II/tables
TID=${SLURM_ARRAY_TASK_ID:-0}
echo "== p2-rerun task $TID in $(pwd) @ $(git rev-parse --short HEAD) on $(hostname) $(date -Is) =="
CHEM=(MUTAG COX2 PTC DHFR NCI1); SOCIAL=(PROTEINS DD IMDB-B IMDB-M NCI109)
AUDITS=(audit_cor32_regularity exp_fps_radius_ablation_pii exp_noninterference_audit exp_certificate_bound_audit exp_domain_inflation exp_pv_joint_feature)
CERT=(Orbit5k MUTAG COX2 DHFR PTC NCI1 PROTEINS DD IMDB-B IMDB-M NCI109)
W=${SLURM_CPUS_PER_TASK:-8}

if [ "$TID" -le 71 ]; then
  S=$((42 + TID / 24)); CFG=$((TID % 24))
  export OMP_NUM_THREADS=1
  PUSH92_SEEDS=$S PUSH92_CONFIGS=$CFG PUSH92_OUT_SUFFIX="_seed${S}_cfg${CFG}" "$PYTHON" -u experiments/exp_orbit5k_push92.py
elif [ "$TID" -eq 72 ]; then
  "$PYTHON" -u experiments/exp_reproduce_orbit5k_90.py | tee results/orbit5k_reproduce_90_rerun.txt
elif [ "$TID" -le 572 ]; then
  if [ "$TID" -le 322 ]; then I=$((TID - 73)); DS=${CHEM[$((I / 50))]}; else I=$((TID - 323)); DS=${SOCIAL[$((I / 50))]}; fi
  S=$(((I % 50) / 10)); F=$((I % 10))
  "$PYTHON" -u experiments/analyze_gamma_filt.py --dataset "$DS" --seeds "$S" --folds "$F" --workers "$W" \
      --out-suffix "_with_mah_seed${S}_part${F}"
elif [ "$TID" -le 578 ]; then
  SCRIPT=${AUDITS[$((TID - 573))]}
  "$PYTHON" -u "experiments/$SCRIPT.py" | tee "results/rerun_${SCRIPT}.txt"
elif [ "$TID" -le 589 ]; then
  D=${CERT[$((TID - 579))]}
  "$PYTHON" -u experiments/cert_firing_valid_fps.py --dataset "$D" | tee "results/paper_II/cert_firing_valid_fps_$D.txt"
elif [ "$TID" -le 594 ]; then
  DS=${CHEM[$((TID - 590))]}
  OUTS=$([ "$DS" = NCI1 ] && echo _with_mah || echo _full46_with_mah)
  "$PYTHON" experiments/merge_seed_files.py --out "$OUTS" "$DS"
  "$PYTHON" -u experiments/exp_outer_test_selected_filt.py --dataset "$DS" --workers "$W" | tee "results/rerun_outer_test_${DS}.txt"
elif [ "$TID" -le 599 ]; then
  DS=${SOCIAL[$((TID - 595))]}
  "$PYTHON" experiments/merge_seed_files.py "$DS"
  "$PYTHON" -u experiments/exp_outer_test_selected_filt.py --dataset "$DS" --workers "$W" | tee "results/rerun_outer_test_${DS}.txt"
else
  echo "unknown task $TID"; exit 1
fi
echo "== done p2-rerun task $TID @ $(date -Is) =="
