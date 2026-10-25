"""Single entry point that retrains every live race model in a fixed order."""
# argparse: stdlib CLI parsing, so the runner adds no dependency.
import argparse
# subprocess: each script runs in its own interpreter, because the scripts are top-level code with no main().
import subprocess
# sys: sys.executable pins the child to this same venv's Python, never a stray global one.
import sys
# pathlib: resolve the repo root from this file, so the runner works from any CWD.
from pathlib import Path

# Repo root = parent of training/; scripts write CWD-relative "models/*.joblib", so they must run from here.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Race name -> live training script; dict order is the run order. 12.py and 17.py are dead experiments, left out on purpose.
RACES = {
    # 6.py writes models/mexico_model.joblib (GBR dict artifact).
    "mexico": "training/6.py",
    # 8-1.py writes models/us_model.joblib (GBR dict artifact).
    "usa": "training/8-1.py",
    # 10-3.py writes models/qatar_model.joblib (bare XGBRegressor).
    "qatar": "training/10-3.py",
    # 12-3.py writes models/abu_dhabi_model.joblib (bare XGBRegressor).
    "abudhabi": "training/12-3.py",
}


# main() keeps argument parsing and execution testable and import-safe.
def main():
    # The parser documents the flags in --help, so nobody has to read this file to use it.
    parser = argparse.ArgumentParser(description="Retrain live F1 race models. WARNING: overwrites models/*.joblib and calls live APIs.")
    # --race is repeatable so one or several races can be retrained without touching the rest.
    parser.add_argument("--race", action="append", choices=list(RACES), help="race to train (repeatable); default: all")
    # --dry-run prints the exact commands, so the plan can be checked without overwriting committed models.
    parser.add_argument("--dry-run", action="store_true", help="print commands only, run nothing")
    # Parse once; argparse already rejected unknown race names.
    args = parser.parse_args()
    # Keep RACES order even when --race is given out of order, so runs are reproducible.
    selected = [name for name in RACES if not args.race or name in args.race]
    # One loop handles both modes, so dry-run prints exactly what a real run executes.
    for name in selected:
        # Same interpreter + script path relative to the repo root (the cwd below).
        cmd = [sys.executable, RACES[name]]
        # Echo every command, so a real run's log shows which race failed.
        print(f"[{name}] cwd={REPO_ROOT} $ {' '.join(cmd)}", flush=True)
        # Dry-run stops here: printing is the whole point of the flag.
        if args.dry_run:
            # Skip execution but keep listing the remaining races.
            continue
        # check=True stops on the first failure, so a broken race never hides behind later successes.
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


# Guard lets the module be imported (e.g. for an import check) without starting training.
if __name__ == "__main__":
    # Run the CLI only when executed as a script.
    main()
