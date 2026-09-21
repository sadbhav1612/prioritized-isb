#!/usr/bin/env bash
# ============================================================================
# Train the P-ISB locomotion grid (arms x seeds) on one GPU, sequentially.
#
# Cells run one at a time: two concurrent 1024-environment sims contend for
# GPU memory and compute. A cell whose result fragment exists is skipped, so
# the script can be interrupted and restarted.
#
# Usage
#   bash pisb_legged/run_local.sh                  # the full 25-cell grid
#   SMOKE=1 bash pisb_legged/run_local.sh          # 10 iterations per cell
#   ARMS="pisb obs" SEEDS="0 6" bash pisb_legged/run_local.sh
#   SINGLE_START=1 bash pisb_legged/run_local.sh   # single-start protocol
#
# RUN THE SMOKE TEST FIRST. It exercises every arm at 10 iterations, which
# catches a broken arm in minutes instead of after the first hour-long cell.
# ============================================================================
set -uo pipefail

LEG_ROOT="${LEG_ROOT:-$HOME/legged_gym}"
cd "$LEG_ROOT" || { echo "ABORT: no $LEG_ROOT"; exit 1; }

# --------------------------------------------------------------- conda
# Works with either anaconda3 or miniconda3.
CONDA_SH=""
for c in "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "$HOME/miniconda3/etc/profile.d/conda.sh"; do
  [ -f "$c" ] && { CONDA_SH="$c"; break; }
done
[ -n "$CONDA_SH" ] || { echo "ABORT: no conda.sh found"; exit 1; }

# `set -u` off across activation: this env's activate.d hook reads
# $MKL_INTERFACE_LAYER before setting it, which is fatal under -u.
set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate pisb_legged || { echo "ABORT: cannot activate pisb_legged"; exit 1; }
set -u

export PYTHONNOUSERSITE=1
export PYTHONPATH="$LEG_ROOT:${PYTHONPATH:-}"

# --------------------------------------------------------------- the grid
TASK="${TASK:-anymal_c_rough}"
NENVS="${NENVS:-1024}"
# train_pisb.py defaults to 200000, at which the
# obs arm re-clusters 200000 states per PPO iteration and effectively stalls --
# in practice obs_s0 sat at iteration 21 for half an hour while vanilla
# finished 2000 on the same node. A cell trained at a different capacity is not
# comparable to one trained with another capacity.
CAPACITY="${CAPACITY:-20000}"
# Bug-fix switches (2026-09-14). Defaults are the FIXED behaviour; the runs
# reported before that date used PRIO_RAW=0 FIFO_UNIFORM=0. TAG keeps variants
# of one arm at one seed in separate directories -- give each variant its own.
PRIO_RAW="${PRIO_RAW:-1}"
FIFO_UNIFORM="${FIFO_UNIFORM:-1}"
TAG="${TAG:-}"
SMOKE="${SMOKE:-0}"
SINGLE_START="${SINGLE_START:-0}"
# Which terrain patch a single-start campaign pins to. -1 -1 is the middle of
# the grid, which is pyramid_stairs DESCENDING (14 cm steps) with a 3 m
# platform: the robot cannot walk off it without falling, so no arm explores
# and the initial-state buffer has only platform poses to resample.
START_ROW="${START_ROW:--1}"
START_COL="${START_COL:--1}"
if [ "$SMOKE" = "1" ]; then
  ITERS="${ITERS:-10}"
  OUT="${OUT:-$LEG_ROOT/pisb_runs_smoke}"
else
  ITERS="${ITERS:-2000}"
  # Overridable so a campaign gets its OWN fragment tree. validate_legged.py
  # scores every fragment under <OUT>/frag, so two campaigns sharing a tree
  # would be validated together.
  OUT="${OUT:-$LEG_ROOT/pisb_runs}"
fi
read -r -a ARMS  <<< "${ARMS:-vanilla random obs cl pisb}"
read -r -a SEEDS <<< "${SEEDS:-0 6 7 8 13}"

mkdir -p "$OUT/frag" "$OUT/logs"
NCELLS=$(( ${#ARMS[@]} * ${#SEEDS[@]} ))

echo "============================================================"
echo "P-ISB locomotion, local GPU"
echo "  host    : $(hostname)"
echo "  gpu     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  env     : $CONDA_PREFIX"
echo "  task    : $TASK   envs=$NENVS   iters=$ITERS"
echo "  grid    : ${#ARMS[@]} arms x ${#SEEDS[@]} seeds = $NCELLS cells"
echo "  out     : $OUT"
[ "$SMOKE" = "1" ] && echo "  MODE    : SMOKE -- 10 iterations, results are NOT for the paper"
echo "============================================================"

# ------------------------------------------------------------- preflight
# Import once up front. A missing package or a broken arm surfaces here in
# seconds rather than after the first cell has burned an hour.
python - <<'PYCHK' || { echo "ABORT: environment incomplete"; exit 1; }
import sys
from isaacgym import gymapi, gymtorch          # MUST precede torch
import torch
import legged_gym, rsl_rl                       # noqa: F401
from pisb_legged.state_buffer import ARMS
from pisb_legged.integration import ISBPPO, ISBOnPolicyRunner, ISBLeggedRobot  # noqa: F401
from initial_buffer.algorithms.projection_buffer import ProjectionBuffer
import inspect
need = {"sampling_strategy", "nr_clusters", "cluster_embedding_dim",
        "nr_mining_samples", "min_timesteps", "n_train_data"}
missing = need - set(inspect.getfullargspec(ProjectionBuffer.__init__).args)
if missing:
    sys.exit("ABORT: initial_buffer is a stale copy, missing %s" % sorted(missing))
if not torch.cuda.is_available():
    sys.exit("ABORT: no CUDA device")
print("  OK   imports | cuda:", torch.cuda.get_device_name(0), "| arms:", ARMS)
PYCHK

# --------------------------------------------------------------- run one
t_start=$(date +%s)
n_done=0; n_skip=0; n_fail=0; i=0
FAILED=()

# Ctrl-C leaves a partial cell behind; say so rather than looking complete.
trap 'echo; echo "INTERRUPTED after $n_done cells. Re-run to resume."; exit 130' INT

for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    i=$((i+1))
    tag="${arm}_s${seed}${TAG}"
    frag="$OUT/frag/${tag}.json"

    if [ -f "$frag" ]; then
      echo "[$i/$NCELLS] skip $tag (fragment exists)"
      n_skip=$((n_skip+1)); continue
    fi

    tmp="$OUT/frag_tmp_${tag}"
    rm -rf "$tmp"

    # CLEAR THE LOG DIRECTORY FIRST. train_pisb.py names it after the cell
    # (logs/rough_anymal_c/<arm>_s<seed>) with NO timestamp, so a re-run appends
    # a second events.out.tfevents.* file beside the first and the aggregator
    # would read a curve stitched from two different runs -- a smoke run and a
    # real one, say. Only reached when the cell is actually about to train, so a
    # completed cell's logs are never touched.
    # Globbed rather than constructed: legged_gym derives the experiment folder
    # from the task's config (anymal_c_rough -> logs/rough_anymal_c), and that
    # mapping is not a string transformation of the task name.
    for ldir in "$LEG_ROOT"/logs/*/"${tag}"; do
      [ -d "$ldir" ] && rm -rf "$ldir"
    done

    echo "[$i/$NCELLS] $(date '+%F %T')  >>> $tag"
    c0=$(date +%s)

    python pisb_legged/train_pisb.py \
        --pisb_arm "$arm" --seed "$seed" --task "$TASK" --headless \
        --num_envs "$NENVS" --max_iterations "$ITERS" \
        --pisb_capacity "$CAPACITY" \
        --pisb_prio_raw "$PRIO_RAW" --pisb_fifo_uniform "$FIFO_UNIFORM" \
        --pisb_tag "$TAG" --pisb_single_start "$SINGLE_START" \
        --pisb_start_row "$START_ROW" --pisb_start_col "$START_COL" \
        --rl_device cuda:0 --sim_device cuda:0 \
        --pisb_out "$tmp" \
        > "$OUT/logs/${tag}.log" 2>&1
    rc=$?
    c1=$(date +%s); dt=$((c1-c0))

    if [ "$rc" -ne 0 ] || [ ! -f "$tmp/fragment.json" ]; then
      echo "        FAILED (rc=$rc) after ${dt}s -- tail of ${tag}.log:"
      tail -15 "$OUT/logs/${tag}.log" | sed 's/^/          /'
      n_fail=$((n_fail+1)); FAILED+=("$tag")
      continue
    fi

    mv "$tmp/fragment.json" "$frag"
    rmdir "$tmp" 2>/dev/null
    n_done=$((n_done+1))

    # ETA from cells actually trained in THIS run, not from skipped ones.
    left=$(( NCELLS - i ))
    eta=$(( dt * left ))
    printf '        done in %dm%02ds   |  %d left  |  eta ~%dh%02dm\n' \
      $((dt/60)) $((dt%60)) "$left" $((eta/3600)) $(((eta%3600)/60))
  done
done

echo
echo "============================================================"
printf 'finished in %dh%02dm   trained=%d skipped=%d failed=%d\n' \
  $(( ($(date +%s)-t_start)/3600 )) $(( (($(date +%s)-t_start)%3600)/60 )) \
  "$n_done" "$n_skip" "$n_fail"
echo "fragments: $(ls "$OUT/frag" 2>/dev/null | wc -l)/$NCELLS  in $OUT/frag"
[ ${#FAILED[@]} -gt 0 ] && echo "FAILED cells: ${FAILED[*]}"
echo "============================================================"
exit $(( n_fail > 0 ? 1 : 0 ))
