"""Phase 12.9B replication bootstrap - writes impl then runs it."""
import subprocess, sys
from pathlib import Path
REPO = Path(r"E:\Orbit (Optimized Research & Behavioral Intelligence Trading)")
parts = ["rep_common", "rep_phase11", "rep_phase12a", "rep_phase12d", "rep_phase12e", "rep_crossphase", "rep_main"]
# Build combined impl
impl_lines = []
for p in parts:
    f = REPO / "scripts" / f"_phase12_9b_{p}.py"
    if f.exists():
        impl_lines.append(f.read_text(encoding="utf-8"))
impl = "\n\n".join(impl_lines)
target = REPO / "scripts" / "_phase12_9b_impl.py"
target.write_text(impl, encoding="utf-8")
print(f"Impl written ({len(impl)} chars)")
subprocess.run([sys.executable, "-u", str(target)], cwd=str(REPO))
