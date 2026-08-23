"""Inspect instrument masters for sector data."""
import json

# Dev universe
with open("configs/instrument_master_dev.json") as f:
    dev = json.load(f)
print("Dev universe sectors:")
sectors = {}
for inst in dev["instruments"]:
    s = inst.get("sector_id", "NONE")
    sectors[s] = sectors.get(s, 0) + 1
for s, c in sorted(sectors.items()):
    print(f"  {s}: {c} instruments")
print(f"Total: {len(dev['instruments'])} instruments")
print()

# Universe-050
with open("configs/instrument_master_universe-050.json") as f:
    u50 = json.load(f)
print("Universe-050 sectors:")
sectors50 = {}
for inst in u50["instruments"]:
    s = inst.get("sector", inst.get("sector_id", "NONE"))
    sectors50[s] = sectors50.get(s, 0) + 1
for s, c in sorted(sectors50.items()):
    print(f"  {s}: {c} instruments")
print(f"Total: {len(u50['instruments'])} instruments")
print()

# Universe-100
with open("configs/instrument_master_universe-100.json") as f:
    u100 = json.load(f)
print("Universe-100 sectors:")
sectors100 = {}
for inst in u100["instruments"]:
    s = inst.get("sector", inst.get("sector_id", "NONE"))
    sectors100[s] = sectors100.get(s, 0) + 1
for s, c in sorted(sectors100.items()):
    print(f"  {s}: {c} instruments")
print(f"Total: {len(u100['instruments'])} instruments")

# Show first 3 entries from each
print("\nDev[0]:", json.dumps(dev["instruments"][0], indent=2))
print("\nU50[0]:", json.dumps(u50["instruments"][0], indent=2))
print("\nU100[0]:", json.dumps(u100["instruments"][0], indent=2))
