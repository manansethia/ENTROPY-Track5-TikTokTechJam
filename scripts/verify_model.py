import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from models.tri_hybrid_detector import MasterEnsembleDetector


def main():
    p = argparse.ArgumentParser(description="Verify instantiated model is below 2B parameters")
    p.add_argument("--config", default="configs/train_config.yaml")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = MasterEnsembleDetector(**cfg["models"])
    report = model.parameter_report()
    print(report)
    if report["total"] >= 2_000_000_000:
        raise SystemExit("FAIL: model is not below the 2B parameter ceiling.")
    print("PASS: total parameters are below 2B.")


if __name__ == "__main__":
    main()
