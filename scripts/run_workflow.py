"""Reproduce the frozen ShopSmart submission from its local source CSV."""
import argparse
import subprocess
import sys

from .audit_intake import ROOT


def main():
    """Use the current Python environment for every stage and stop on any failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    source = parser.parse_args().source
    for stage in ["audit_intake", "prepare_split", "development_eda", "compare_models",
                  "select_model", "evaluate_holdout", "finalize_dual"]:
        command = [sys.executable, "-m", f"scripts.{stage}"]
        if stage == "audit_intake":
            command.extend(["--source", source])
        print(f"Running {stage}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
