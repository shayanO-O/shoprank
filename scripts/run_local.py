#!/usr/bin/env python3
"""
ShopRank Local Serving & Router Launcher
Launches Champion (port 8000), Challenger (port 8001), and Traffic Router (port 8080) concurrently.
Non-blocking pre-flight checks guarantee ports are available without hanging.
Press Ctrl+C to stop all services.
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

def can_bind(port: int) -> bool:
    """Instantaneous test whether the port can be bound."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()

def ensure_port_available(port: int, force: bool = True) -> bool:
    """Non-blocking port availability check and cleanup."""
    if can_bind(port):
        return True

    print(f"⚠️  Port {port} is occupied by another process.")
    if force:
        print(f"   Terminating lingering process on port {port}...")
        try:
            res = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True, timeout=1.0)
            for pid in res.stdout.strip().split():
                if pid.isdigit() and int(pid) != os.getpid():
                    os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass

        try:
            subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1.0)
        except Exception:
            pass

        time.sleep(0.3)

    if not can_bind(port):
        print(f"❌ Error: Port {port} could not be freed!")
        print(f"   Run: fuser -k {port}/tcp")
        print(f"   Or check running Docker containers: docker compose down")
        return False

    print(f"✅ Port {port} released successfully.")
    return True

def main():
    parser = argparse.ArgumentParser(description="ShopRank Local Serving Stack Launcher")
    parser.add_argument("--champion-only", action="store_true", help="Launch only Champion (8000) and Router (8080)")
    parser.add_argument("--champion-port", type=int, default=8000, help="Champion port (default: 8000)")
    parser.add_argument("--challenger-port", type=int, default=8001, help="Challenger port (default: 8001)")
    parser.add_argument("--router-port", type=int, default=8080, help="Router port (default: 8080)")
    args = parser.parse_args()

    python_bin = sys.executable
    print("=" * 75)
    print("Starting ShopRank Local Serving Stack...")
    print("=" * 75)

    ports_to_check = [args.champion_port, args.router_port]
    if not args.champion_only:
        ports_to_check.append(args.challenger_port)

    for p in ports_to_check:
        if not ensure_port_available(p):
            sys.exit(1)

    procs = []

    # 1. Start Champion Serving
    total_services = 2 if args.champion_only else 3
    print(f"\n[1/{total_services}] Launching Champion Serving on http://127.0.0.1:{args.champion_port} (MODEL_VERSION=champion)...")
    champ_env = os.environ.copy()
    champ_env["PYTHONPATH"] = str(root_dir)
    champ_env["MODEL_VERSION"] = "champion"
    champ_cmd = [
        python_bin, "-m", "uvicorn",
        "src.serving.app:app",
        "--host", "0.0.0.0",
        "--port", str(args.champion_port)
    ]
    champ_proc = subprocess.Popen(champ_cmd, cwd=root_dir, env=champ_env)
    procs.append(("Champion", champ_proc))

    # 2. Start Challenger Serving (if not champion-only)
    chal_proc = None
    if not args.champion_only:
        print(f"[2/3] Launching Challenger Serving on http://127.0.0.1:{args.challenger_port} (MODEL_VERSION=challenger)...")
        chal_env = os.environ.copy()
        chal_env["PYTHONPATH"] = str(root_dir)
        chal_env["MODEL_VERSION"] = "challenger"
        chal_cmd = [
            python_bin, "-m", "uvicorn",
            "src.serving.app:app",
            "--host", "0.0.0.0",
            "--port", str(args.challenger_port)
        ]
        chal_proc = subprocess.Popen(chal_cmd, cwd=root_dir, env=chal_env)
        procs.append(("Challenger", chal_proc))

    # 3. Start Traffic Router
    step_num = 2 if args.champion_only else 3
    print(f"[{step_num}/{total_services}] Launching Traffic Router on http://127.0.0.1:{args.router_port}...")
    router_env = os.environ.copy()
    router_env["PYTHONPATH"] = str(root_dir)
    router_env["CHAMPION_URL"] = f"http://localhost:{args.champion_port}"
    if chal_proc:
        router_env["CHALLENGER_URL"] = f"http://localhost:{args.challenger_port}"
        router_env["CANARY_WEIGHT"] = "10"
        router_env["SHADOW_ENABLED"] = "true"
    else:
        router_env["CHALLENGER_URL"] = f"http://localhost:{args.champion_port}"
        router_env["CANARY_WEIGHT"] = "0"
        router_env["SHADOW_ENABLED"] = "false"

    router_cmd = [
        python_bin, "-m", "uvicorn",
        "src.serving.router:router_app",
        "--host", "0.0.0.0",
        "--port", str(args.router_port)
    ]
    router_proc = subprocess.Popen(router_cmd, cwd=root_dir, env=router_env)
    procs.append(("Router", router_proc))

    # Clean shutdown handler
    def shutdown(sig, frame):
        print("\nStopping ShopRank services...")
        for name, p in reversed(procs):
            p.terminate()
        for name, p in procs:
            p.wait()
        print("All services stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Startup grace period for loading bundles and feature store
    time.sleep(3)
    print("\n" + "=" * 75)
    print("ShopRank Stack is RUNNING!")
    print(f"  - Router Gateway:   http://localhost:{args.router_port}/recommend")
    print(f"  - Champion Direct:  http://localhost:{args.champion_port}/recommend (model: champion)")
    if chal_proc:
        print(f"  - Challenger Direct:http://localhost:{args.challenger_port}/recommend (model: challenger)")
    print(f"  - Swagger Docs:     http://localhost:{args.router_port}/docs")
    print("=" * 75)
    print("\nIn another terminal, run the simulator:")
    print(f"  python src/serving/simulator.py --endpoint http://localhost:{args.router_port}/recommend --n-requests 500\n")
    print("Press Ctrl+C in this terminal to stop the stack.")

    try:
        while True:
            time.sleep(1)
            for name, p in procs:
                if p.poll() is not None:
                    print(f"\n❌ Error: {name} process exited with code {p.returncode}")
                    shutdown(None, None)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(None, None)

if __name__ == "__main__":
    main()
