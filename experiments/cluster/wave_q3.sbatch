#!/bin/bash
#SBATCH --job-name=p2-certvalid
#SBATCH --account=REDACTED_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=cluster/logs/waveq3_%A_%a.out
#SBATCH --error=cluster/logs/waveq3_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=REDACTED
# Q3: valid certificate firing on the FPS embedding, 11 data sets (experiments/cert_firing_valid_fps.py)
#   PLEB_HOME=$HOME/pleb2 sbatch --export=ALL,PLEB_HOME --array=0-10 cluster/wave_q3.sbatch
set -euo pipefail
module purge; module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/pleb-env/bin/python"; export PATH="$HOME/.conda/envs/pleb-env/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "${PLEB_HOME:-$HOME/pleb}"; mkdir -p cluster/logs results/paper_II
DS=(Orbit5k MUTAG COX2 DHFR PTC NCI1 PROTEINS DD IMDB-B IMDB-M NCI109)
D=${DS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "== p2-certvalid $D in $(pwd) @ $(git rev-parse --short HEAD) $(date -Is) =="
"$PYTHON" -u experiments/cert_firing_valid_fps.py --dataset "$D" | tee "results/paper_II/cert_firing_valid_fps_$D.txt"
echo "== done $D @ $(date -Is) =="
