#!/bin/bash
#SBATCH --job-name=p2-outer-social
#SBATCH --account=REDACTED_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=cluster/logs/waveq2b_%A_%a.out
#SBATCH --error=cluster/logs/waveq2b_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=REDACTED
# Q2b stage 2: merge the per-seed analysis files, then outer-fold test accuracy at the selected
# filtration for the lower block.  Submit with --dependency=afterok:<Q2b analyze job>.
#   PLEB_HOME=$HOME/pleb2 sbatch --export=ALL,PLEB_HOME --array=0-4 --dependency=afterok:138705196 cluster/wave_q2b_outer.sbatch
set -euo pipefail
module purge; module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/pleb-env/bin/python"; export PATH="$HOME/.conda/envs/pleb-env/bin:$PATH"
cd "${PLEB_HOME:-$HOME/pleb}"; mkdir -p cluster/logs
DS=(PROTEINS DD IMDB-B IMDB-M NCI109); D=${DS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "== p2-outer-social $D in $(pwd) @ $(git rev-parse --short HEAD) $(date -Is) =="
"$PYTHON" experiments/merge_seed_files.py "$D"
"$PYTHON" -u experiments/exp_outer_test_selected_filt.py --dataset "$D" --workers "${SLURM_CPUS_PER_TASK:-16}" \
    | tee "results/rerun_outer_test_${D}.txt"
echo "== done $D @ $(date -Is) =="
