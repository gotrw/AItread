"""
MIRA_trader_C – Launch web dashboard.

Usage:
    cd MIRA_trader_C
    python web/run.py [--host 0.0.0.0] [--port 8080] [--reload]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="MIRA_trader_C Web Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    print(f"\n  🚀 MIRA trader C Dashboard")
    print(f"  ─────────────────────────────────────────")
    print(f"  URL   : http://localhost:{args.port}")
    print(f"  Mode  : {'development (reload)' if args.reload else 'production'}")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(
        "web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(Path(__file__).parent.parent)] if args.reload else None,
        log_level="info",
    )


if __name__ == "__main__":
    main()
