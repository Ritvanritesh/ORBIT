from orbit.ml.dataset import assemble_datasets
from orbit.ml.features import FeatureSnapshot
import polars as pl

# Minimal test
records = pl.DataFrame({
    'instrument_id': ['INS-000001'],
    'decision_session': ['2020-01-02'],
    'decision_time': ['2020-01-02T16:00:00'],
    'ret_10': [0.01],
    'ret_20': [0.015],
    'ret_30': [0.02],
    'sma_ratio_5_30': [1.001],
    'sma_ratio_15_40': [1.0],
    'vol_10': [0.2],
    'vol_30': [0.25],
    'log_dv_med_20': [8.0],
})

fs_snap = FeatureSnapshot(
    feature_set_id='FS-12B-A',
    feature_set_version='v1',
    feature_refs=sorted(['ret_10', 'ret_20', 'ret_30', 'sma_ratio_5_30', 'sma_ratio_15_40', 'vol_10', 'vol_30', 'log_dv_med_20']),
    data_refs=sorted(['ret_10', 'ret_20', 'ret_30', 'sma_ratio_5_30', 'sma_ratio_15_40', 'vol_10', 'vol_30', 'log_dv_med_20']),
    records=records,
})

# Minimal label
label_records = pl.DataFrame({
    'instrument_id': ['INS-000001'],
    'decision_session': ['2020-01-02'],
    'decision_time': ['2020-01-02T16:00:00'],
    'outcome_value': [0.05],
    'outcome_status': ['available'],
})

lab_snap = type('obj', (object,), {'records': label_records})()

ds = assemble_datasets(fs_snap, lab_snap, feature_names=['ret_10', 'ret_20', 'ret_30', 'sma_ratio_5_30', 'sma_ratio_15_40', 'vol_10', 'vol_30', 'log_dv_med_20'])
print('Dataset assembled successfully')
print('Train:', ds['train'][0].shape)
print('Test:', ds['test'][0].shape)
meta_test = ds['test'][3]
print('Meta test columns:', meta_test.columns)
print(meta_test)