#!/bin/bash
# Phase 1 pipeline: final sweep -> pick config -> main Dots run -> baselines.
# Resumable: each step is skipped when its result file already exists.
set -u
cd "$(dirname "$0")"
T="python -u tune.py --layer 15 --opt adam"
[ -f runs/tune_L15_adam_p90lr06.json ] || $T --lrs 0.06 --protect 0.9 --out _p90lr06 > t4.txt 2>&1
[ -f runs/tune_L15_adam_p95.json ] || $T --lrs 0.03 --protect 0.95 --out _p95 > t5.txt 2>&1
[ -f runs/tune_L15_adam_p90s120.json ] || $T --lrs 0.03 --protect 0.9 --steps 120 --out _p90s120 > t6.txt 2>&1
ARGS=$(python pick_best.py)
echo "BEST: $ARGS"
[ -f runs/dots_main.json ] || python -u run_dots.py $ARGS --tag main >> main_log.txt 2>&1
[ -f runs/baseline_lora.json ] || python -u run_baselines.py --method lora --lr 3e-4 > lora_log.txt 2>&1
[ -f runs/baseline_fullft.json ] || python -u run_baselines.py --method fullft --lr 2e-5 > fullft_log.txt 2>&1
echo PIPELINE_DONE
