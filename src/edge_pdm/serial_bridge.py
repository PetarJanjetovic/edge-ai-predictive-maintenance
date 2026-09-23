import json
import time

import serial

from .simulator import post_json


def run_serial_bridge(port: str, url: str, baud: int = 115200) -> None:
    with serial.Serial(port, baud, timeout=1) as connection:
        print(f"forwarding {port} -> {url}")
        while True:
            raw = connection.readline().decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
                payload.setdefault("timestamp", time.time())
                post_json(url, payload)
                print(raw)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"ignored line: {exc}")

