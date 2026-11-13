#!/bin/bash
#SBATCH --job-name=p2-rerun
#SBATCH --account=REDACTED_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=cluster/logs/waveq_%A_%a.out
#SBATCH --error=cluster/logs/waveq_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=REDACTED
#
# Paper II re-run on the definition-faithful code (notes/paper-II-code-review-2026-11-08.md).
# Run from a clean clone (PLEB_HOME=$HOME/pleb2): results/ is empty there, so no embedding cache
# built by the old landmark code can be reused; data/ is a symlink to the shared diagram caches.
#
#   0-2    exp_orbit5k_push92.py, one outer-CV seed per task (42,43,44); per-fold CSVs      [Q1]
#   3      exp_reproduce_orbit5k_90.py  (90.4 baseline table)                                   [Q1]
#   4-8    analyze_gamma_filt.py  MUTAG COX2 PTC DHFR NCI1, seeds 0-4 (selectors + inner CV)  [Q2]
#   9-14   audits: cor32, fps_radius_ablation, noninterference, certificate_bound,
#          domain_inflation, pv_joint_feature                                                 [Q4]
#   15-19  exp_outer_test_selected_filt.py  per dataset  -- submit with --dependency=afterok:<Q2 job> [Q2]
#   20-44  analyze_gamma_filt.py  PROTEINS DD IMDB-B IMDB-M NCI109 x seeds 0-4 (FPS lower block) [Q2b]
#
#   PLEB_HOME=$HOME/pleb2
#   A=$(sbatch --parsable --export=ALL,PLEB_HOME --array=0-3 --cpus-per-task=10 --mem=96G cluster/wave_q.sbatch)
#   B=$(sbatch --parsable --export=ALL,PLEB_HOME --array=4-8 cluster/wave_q.sbatch)
#   C=$(sbatch --parsable --export=ALL,PLEB_HOME --array=9-14 --cpus-per-task=8 --mem=32G --time=12:00:00 cluster/wave_q.sbatch)
#   D=$(sbatch --parsable --export=ALL,PLEB_HOME --array=15-19 --dependency=afterok:$B cluster/wave_q.sbatch)
#   E=$(sbatch --parsable --export=ALL,PLEB_HOME --array=20-44 --mem=192G --time=12:00:00 cluster/wave_q.sbatch)
#
set -euo pipefail
module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/pleb-env/bin/python"
export PATH="$HOME/.conda/envs/pleb-env/bin:$PATH"
cd "${PLEB_HOME:-$HOME/pleb}"
mkdir -p cluster/logs results/orbit5k_final_sweep results/filtration_grid/analysis results/paper_II/tables
TID=${SLURM_ARRAY_TASK_ID:-0}
echo "== p2-rerun task $TID in $(pwd) @ $(git rev-parse --short HEAD) on $(hostname) $(date -Is) =="

CHEM=(MUTAG COX2 PTC DHFR NCI1)
SOCIAL=(PROTEINS DD IMDB-B IMDB-M NCI109)
AUDITS=(audit_cor32_regularity exp_fps_radius_ablation_pii exp_noninterference_audit exp_certificate_bound_audit exp_domain_inflation exp_pv_joint_feature)

if [ "$TID" -le 2 ]; then
  S=$((42 + TID))
  PUSH92_SEEDS=$S PUSH92_OUT_SUFFIX="_seed$S" "$PYTHON" -u experiments/exp_orbit5k_push92.py
elif [ "$TID" -eq 3 ]; then
  "$PYTHON" -u experiments/exp_reproduce_orbit5k_90.py | tee results/orbit5k_reproduce_90_rerun.txt
elif [ "$TID" -le 8 ]; then
  DS=${CHEM[$((TID - 4))]}
  SUF=$([ "$DS" = NCI1 ] && echo _with_mah || echo _full46_with_mah)   # the names exp_outer_test_selected_filt expects
  "$PYTHON" -u experiments/analyze_gamma_filt.py --dataset "$DS" --seeds 0 1 2 3 4 \
      --workers "${SLURM_CPUS_PER_TASK:-16}" --out-suffix "$SUF"
elif [ "$TID" -le 14 ]; then
  SCRIPT=${AUDITS[$((TID - 9))]}
  "$PYTHON" -u "experiments/$SCRIPT.py" | tee "results/rerun_${SCRIPT}.txt"
elif [ "$TID" -le 19 ]; then
  DS=${CHEM[$((TID - 15))]}
  "$PYTHON" -u experiments/exp_outer_test_selected_filt.py --dataset "$DS" --workers "${SLURM_CPUS_PER_TASK:-16}" \
      | tee "results/rerun_outer_test_${DS}.txt"
elif [ "$TID" -le 44 ]; then
  I=$((TID - 20)); DS=${SOCIAL[$((I / 5))]}; S=$((I % 5))
  "$PYTHON" -u experiments/analyze_gamma_filt.py --dataset "$DS" --seeds "$S" \
      --workers "${SLURM_CPUS_PER_TASK:-16}" --out-suffix "_with_mah_seed$S"
else
  echo "unknown task $TID"; exit 1
fi
echo "== done p2-rerun task $TID @ $(date -Is) =="
