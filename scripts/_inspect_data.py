import polars as pl
import json

# Check DS-EXP-050
df050 = pl.read_parquet(r"data\normalized\market\yahoo_chart_api\DS-EXP-050\bars.parquet")
print("=== DS-EXP-050 ===")
print(f"Shape: {df050.shape}")
print(f"Columns: {df050.columns}")
print(f"Schema: {df050.schema}")
print(f"Date range: {df050['trade_date'].min()} to {df050['trade_date'].max()}")
print(f"Instruments: {df050['instrument_id'].n_unique()}")
print()

# Check DS-EXP-100
df100 = pl.read_parquet(r"data\normalized\market\yahoo_chart_api\DS-EXP-100\bars.parquet")
print("=== DS-EXP-100 ===")
print(f"Shape: {df100.shape}")
print(f"Columns: {df100.columns}")
print(f"Date range: {df100['trade_date'].min()} to {df100['trade_date'].max()}")
print(f"Instruments: {df100['instrument_id'].n_unique()}")
print()

# Check macro data
df_macro = pl.read_parquet(r"data\normalized\macro\fred_csv\DS-000003\series.parquet")
print("=== DS-000003 (Macro) ===")
print(f"Shape: {df_macro.shape}")
print(f"Columns: {df_macro.columns}")
print(f"Schema: {df_macro.schema}")
# Show unique series names
if "series_id" in df_macro.columns:
    print(f"Series IDs: {df_macro['series_id'].unique().to_list()}")
print()

# Check benchmark
df_bench = pl.read_parquet(r"data\normalized\benchmark\BENCH-001\bars.parquet")
print("=== BENCH-001 ===")
print(f"Shape: {df_bench.shape}")
print(f"Columns: {df_bench.columns}")
print(f"Date range: {df_bench['trade_date'].min()} to {df_bench['trade_date'].max()}")
print()

# Show sample of DS-EXP-050
print("=== DS-EXP-050 Sample (first 3 rows) ===")
print(df050.head(3))
print()

# Show sample of DS-EXP-100
print("=== DS-EXP-100 Sample (first 3 rows) ===")
print(df100.head(3))
print()

# Show macro sample
print("=== DS-000003 Sample (first 5 rows) ===")
print(df_macro.head(5))
