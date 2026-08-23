"""Phase 12.9A Audit Bootstrap - generates and runs the full audit implementation."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "_phase12_9a_audit_impl.py"

# Read the template and write the impl
TEMPLATE = Path(__file__).resolve().parent / "phase12_9a_template.py"
SCRIPT.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
print(f"Impl written: {SCRIPT}")
subprocess.run([sys.executable, "-u", str(SCRIPT)], cwd=str(REPO))
