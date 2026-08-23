"""Inspect SEC EDGAR fundamental datasets."""
import json, os

for ds in ['DS-000002', 'DS-000005']:
    d = f'data/normalized/fundamentals/sec_edgar_companyfacts/{ds}/'
    files = os.listdir(d) if os.path.isdir(d) else []
    print(f'\n=== {ds} ===')
    print(f'Files: {len(files)}')
    if files:
        with open(os.path.join(d, files[0])) as f:
            data = json.load(f)
        print(f'Keys: {list(data.keys())[:15]}')
        if 'company' in data:
            cname = data['company']
            # Truncate if too long
            print(f'Company: {cname if len(cname) < 50 else cname[:50] + \"...\"}')
        if 'facts' in data:
            facts = data['facts']
            fact_keys = list(facts.keys())[:20] if isinstance(facts, dict) else 'N/A'
            print(f'Fact keys sample: {fact_keys}')
            # Count total facts
            print(f'Total facts: {len(facts)}')
    else:
        print('  No files in this dataset')