from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ARTIFACT_DIR, DATA_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edge-pdm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a synthetic labeled dataset")
    generate.add_argument("--samples-per-class", type=int, default=500)
    generate.add_argument("--output", type=Path, default=DATA_DIR / "processed" / "synthetic.npz")
    generate.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train", help="train, evaluate, and export the edge model")
    train.add_argument("--input", type=Path, default=DATA_DIR / "processed" / "synthetic.npz")
    train.add_argument("--seed", type=int, default=42)

    subparsers.add_parser("evaluate", help="print saved evaluation metrics")

    dashboard = subparsers.add_parser("dashboard", help="serve the live dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8000)

    simulate = subparsers.add_parser("simulate", help="stream predicted synthetic telemetry")
    simulate.add_argument("--url", default="http://127.0.0.1:8000/api/telemetry")
    simulate.add_argument("--interval", type=float, default=0.5)

    serial = subparsers.add_parser("serial", help="bridge ESP32 JSON serial output to dashboard")
    serial.add_argument("--port", required=True)
    serial.add_argument("--url", default="http://127.0.0.1:8000/api/telemetry")
    serial.add_argument("--baud", type=int, default=115200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate":
        from .data import save_dataset
        from .simulator import MachineSignalGenerator
        generator = MachineSignalGenerator(args.seed)
        windows, labels = generator.dataset(args.samples_per_class)
        save_dataset(args.output, windows, labels)
        print(f"saved {len(labels)} windows to {args.output}")
    elif args.command == "train":
        from .data import load_dataset
        from .model import save_bundle, train
        windows, labels = load_dataset(args.input)
        model, metrics = train(windows, labels, args.seed)
        save_bundle(model, metrics)
        print(json.dumps(metrics, indent=2))
    elif args.command == "evaluate":
        path = ARTIFACT_DIR / "metrics.json"
        print(path.read_text() if path.exists() else "No metrics found. Run training first.")
    elif args.command == "dashboard":
        import uvicorn
        uvicorn.run("edge_pdm.dashboard:app", host=args.host, port=args.port)
    elif args.command == "simulate":
        from .simulator import run_telemetry_simulator
        run_telemetry_simulator(args.url, args.interval)
    elif args.command == "serial":
        from .serial_bridge import run_serial_bridge
        run_serial_bridge(args.port, args.url, args.baud)


if __name__ == "__main__":
    main()

