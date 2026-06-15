"""Run an experimental backend for time-step stride animation tests."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def main() -> None:
    """Start a second backend instance without stopping the main 8090 service."""
    parser = argparse.ArgumentParser(description="Run the UFM stride backend service.")
    parser.add_argument(
        "--host",
        default=os.getenv("UFM_STRIDE_HOST", "127.0.0.1"),
        help="Bind host. Defaults to UFM_STRIDE_HOST or 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("UFM_STRIDE_PORT", "8091")),
        help="Bind port. Defaults to UFM_STRIDE_PORT or 8091.",
    )
    args = parser.parse_args()

    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
