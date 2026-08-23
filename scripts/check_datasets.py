"""Quick verification of acquired datasets."""
import polars as pl
from orbit.ingestion.paths import normalized_dir

for snap_id in ["DS-000004", "DS-000006"]:
    bars_path = normalized_dir("market", "yahoo_chart_api", snap_id) / "bars.parquet"
    if bars_path.exists():
        bars = pl.read_parquet(bars_path)
        n_inst = bars["instrument_id"].n_unique()
        n_sess = bars["trade_date"].n_unique()
        symbols = sorted(bars["symbol"].unique().to_list())
        print(f"\n{snap_id}:")
        print(f"  Rows: {bars.height}")
        print(f"  Instruments: {n_inst}")
        print(f"  Sessions: {n_sess}")
        print(f"  Date range: {bars['trade_date'].min()} to {bars['trade_date'].max()}")
        print(f"  Symbols: {symbols}")
    else:
        print(f"\n{snap_id}: NOT FOUND")
