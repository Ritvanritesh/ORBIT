"""Inspect SEC EDGAR fundamental data."""
import json, os
from pathlib import Path

fund_dir = Path('data/normalized/fundamentals/sec_edgar_companyfacts')
if fund_dir.is_dir():
    entries = sorted(fund_dir.glob('*.json'))
    print(f'Files in fundamentals dir: {len(entries)}')
    
    # Check a sample file
    if entries:
        sample = entries[0]
        with open(sample) as f:
            data = json.load(f)
        print(f'Sample file: {sample.name}')
        print(f'Keys: {list(data.keys())[:15]}')
        if 'company' in data:
            print(f'Company: {data["company"]}')
        if 'facts' in data:
            facts = data['facts']
            fact_keys = list(facts.keys())[:20] if isinstance(facts, dict) else 'N/A'
            print(f'Fact keys sample: {fact_keys}')
else:
    print('directory not found')