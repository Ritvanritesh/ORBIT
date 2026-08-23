"""Create synthetic fundamental data for Phase 12B."""
import json, os, random
from pathlib import Path
from datetime import date, timedelta

random.seed(42)
fund_dir = Path('data/normalized/fundamentals/sec_edgar_companyfacts')
ds_dir = fund_dir / 'DS-EXP-050'
ds_dir.mkdir(parents=True, exist_ok=True)

base_date = date(1998, 1, 1)

for i in range(1, 51):
    ticker = f'INS-00{i:04d}'
    # Generate realistic-ish fundamental data
    eps = round(random.uniform(-2, 5), 2)
    bvps = round(random.uniform(10, 200), 2)
    rev = round(random.uniform(100, 10000), 2)
    ni = round(random.uniform(-50, 500), 2)
    ta = round(random.uniform(50, 5000), 2)
    eq = round(random.uniform(20, 2000), 2)
    oi = round(random.uniform(-100, 500), 2)
    td = round(random.uniform(0, 1000), 2)
    ca = round(random.uniform(0, 500), 2)
    cl = round(random.uniform(0, 300), 2)
    
    # Filing date - historical, up to 28 years back
    days_offset = random.randint(0, 365*28)
    filing_date = base_date + timedelta(days=days_offset)
    period_end = date(1997, 12, 31) + timedelta(days=random.randint(0, 365))
    
    company_data = {
        "cik": str(12345 + i),
        "ticker": ticker,
        "company": {"name": f'Company {ticker}'},
        "filing_date": filing_date.isoformat(),
        "period_end_date": period_end.isoformat(),
        "version": "v1.0.0",
        "facts": {
            "eps": eps,
            "revTWelveMonths": rev,
            "ib": oi,
            "ni": ni,
            "at": ta,
            "seq": eq,
            "oancf": round(random.uniform(-100, 500), 2),
            "dvt": td,
            "currassets": ca,
            "currli": cl,
        }
    }
    
    filepath = ds_dir / f'{ticker}.json'
    with open(filepath, 'w') as f:
        json.dump(company_data, f, indent=2)

count = len(list(ds_dir.glob('*.json')))
print(f'Created {count} fundamental data files for DS-EXP-050')