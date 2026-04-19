import subprocess
import itertools
import os
import sys
from pathlib import Path

# ================= PARAMS =================
PYTHON_EXE = r"d:/inno/adv_ml_project/randar/.venv/Scripts/python.exe"
TRAIN_SCRIPT = r"d:/inno/adv_ml_project/randar/train_c2i.py"


MAX_ITERS = 5000

ECE_THRESHOLDS = [0.03, 0.05, 0.07, 0.10]
MAX_SHUFFLE_RATIOS = [0.3, 0.5, 0.7]

RESULTS_DIR = "results"
# ==============================================

def run_experiment(ece_thresh: float, shuffle_ratio: float, idx: int, total: int):
    exp_name = f"adaptive_tuning_ece{ece_thresh}_shuffle{shuffle_ratio}"
    
    cmd = [
        PYTHON_EXE,
        TRAIN_SCRIPT,
        "--max-iters", str(MAX_ITERS),
        "--ece-threshold", str(ece_thresh),
        "--max-shuffle-ratio", str(shuffle_ratio),
        "--exp_name", exp_name
    ]
    
    
    print(f"\n{'='*60}")
    print(f"[{idx}/{total}] Launching: {exp_name}")
    print(f"Params: ece_threshold={ece_thresh}, max_shuffle_ratio={shuffle_ratio}")
    print(f"Dir : {RESULTS_DIR}/{exp_name}")
    print(f"{'='*60}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished : {exp_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error {exp_name} | Code: {e.returncode}")
    except KeyboardInterrupt:
        print("\n Stopped by user.")
        sys.exit(1)

def main():
    combinations = list(itertools.product(ECE_THRESHOLDS, MAX_SHUFFLE_RATIOS))
    total = len(combinations)
    
    print(f" Combinations: {total}")
    print(" Launching.. \n")
    
    for i, (ece, shuffle) in enumerate(combinations, 1):
        run_experiment(ece, shuffle, i, total)
        
    print("\nAll done!")

if __name__ == "__main__":
    main()