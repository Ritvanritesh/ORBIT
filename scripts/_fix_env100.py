"""Fix p() calls in phase12a_env100.py."""
from pathlib import Path

p = Path("scripts/phase12a_env100.py")
c = p.read_text(encoding="utf-8")

# Replace all p("...", end=" ") with print("...", end=" ", flush=True)
import re
# Match p("...", end=" ") and p(f"...", end=" ")
c = c.replace('p("  Market...", end=" ")', 'print("  Market...", end=" ", flush=True)')
c = c.replace('p("  Sector...", end=" ")', 'print("  Sector...", end=" ", flush=True)')
c = c.replace('p("  Cross-sectional...", end=" ")', 'print("  Cross-sectional...", end=" ", flush=True)')
c = c.replace('p(f"    {fs_id} (+{name})...", end=" ")', 'print(f"    {fs_id} (+{name})...", end=" ", flush=True)')
c = c.replace('p("    FS-104 (+all)...", end=" ")', 'print("    FS-104 (+all)...", end=" ", flush=True)')
c = c.replace('p(f"  [{done}/{total}] {fam}+{fs_id} ({elapsed:.0f}s, ETA {eta:.0f}s)", end=" ")', 'print(f"  [{done}/{total}] {fam}+{fs_id} ({elapsed:.0f}s, ETA {eta:.0f}s)", end=" ", flush=True)')

p.write_text(c, encoding="utf-8")
print("Fixed phase12a_env100.py")
