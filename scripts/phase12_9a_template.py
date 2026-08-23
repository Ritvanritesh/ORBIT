"""Phase 12.9A Audit Template - generates the full audit."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "benchmarks"

# Read the implementation and run it
impl = Path(REPO / "scripts" / "_phase12_9a_audit_impl.py")
subprocess.run([sys.executable, "-u", "-c", impl.read_text(encoding="utf-8")],
               cwd=str(REPO))
