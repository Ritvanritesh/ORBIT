import json
import sys
sys.path.insert(0, ".")
from orbit.ml.phase12b_fundamentals import load_sec_edgar_companyfacts

identity_map = {}
for snapshot_id in ['DS-EXP-050', 'DS-EXP-100']:
    df = load_sec_edgar_companyfacts(snapshot_id)
    mappings = []
    for row in df.iter_rows(named=True):
        mappings.append({
            'instrument_id': row.get('ticker', 'N/A'),
            'cik': row.get('cik', 'N/A'),
            'ticker': row.get('ticker', 'N/A'),
            'company': row.get('company', 'N/A'),
            'filing_date': str(row.get('filing_date', 'N/A')),
            'period_end_date': str(row.get('period_end_date', 'N/A')),
        })
    identity_map[snapshot_id] = mappings

with open('benchmarks/phase12b_identity_mapping.json', 'w') as f:
    json.dump({
        'phase': '12B',
        'report_type': 'identity_mapping',
        'created_at': '2026-08-22',
        'data_type': 'synthetic',
        'note': 'All CIKs and ticker mappings are synthetic - not real SEC identities',
        'mappings': identity_map,
    }, f, indent=2, default=str)
print('Identity mapping saved')
print('DS-EXP-050:', len(identity_map.get('DS-EXP-050', [])), 'mappings')
print('DS-EXP-100:', len(identity_map.get('DS-EXP-100', [])), 'mappings')
