import json

for env_id in ["ENV-12B-050", "ENV-12B-100"]:
    fname = f"benchmarks/phase12b_{env_id}_results.json"
    try:
        with open(fname) as f:
            data = json.load(f)
        print(f"\n{env_id}:")
        print(f"  n_successful: {data.get('n_successful')}")
        print(f"  n_blocked: {data.get('n_blocked')}")
        print(f"  pit_compliant: {data.get('pit_compliant')}")
        print(f"  pit_result: {data.get('pit_result')}")
        ok = [r for r in data.get('results', []) if 'error' not in r]
        blocked = [r for r in data.get('results', []) if r.get('blocked')]
        print(f"  successful experiments: {len(ok)}")
        print(f"  blocked experiments: {len(blocked)}")
        for r in ok:
            ic = r.get('metrics', {}).get('oos_ic')
            print(f"    {r['experiment_id']}: IC={ic}")
    except FileNotFoundError:
        print(f"\n{env_id}: not yet saved")
