#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# load
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import lightgbm as lgb
import os, glob, ast
paths = glob.glob('stage1_feather/*/*.feather')
test_path = 'stage1_feather/test'
train_path = 'stage1_feather/train'

for path in paths:
    df = pd.read_feather(path).sort_values('LogTime', ascending=True)
    l1 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    ratio = len(df) / l1
    df['datetime'] = pd.to_datetime(df['LogTime'], unit='s')
    cat_col = [
        'BankId', 'BankgroupId', 'ChannelId', 'CpuId', 'DimmId', 'RankId',
        'deviceID', 'ColumnId', 'RowId'
    ]
    parity_col = [
        'bit_cnt', 'dq_cnt', 'burst_cnt', 'max_dq_interval',
        'max_burst_interval'
    ]
    df['day'] = df['datetime'].dt.dayofyear
    df['diff'] = df['LogTime'].diff()
    def func(x):
        if pd.isna(x):
            return None, None, None, None, None

        x = bin(int(x))[2:].zfill(32)

        ar = [x[i:i + 4].count("1") for i in range(0, 32, 4)]
        ind = [i for i, v in enumerate(ar) if v > 0]
        d = (ind[-1] - ind[0] if ind else 0)
        inds = [
            i for i, v in enumerate([x[i::4].count("1") for i in range(4)])
            if v > 0
        ]
        c = (inds[-1] - inds[0] if inds else 0)

        return bin_parity.count("1"), len(inds), len(ind), c, d

    feat = df['RetryRdErrLogParity'].apply(func)
    feat = feat.apply(pd.Series)
    feat.columns = parity_col
    df = pd.concat([df, feat], axis=1)
    params = {
        'LogTime': 'count',
        'datetime': 'first',
        'diff': ['min', 'std'],
        'storms': ['sum', 'mean']
    }
    for col in cat_col:
        params[col] = 'nunique'
    for col in parity_col:
        params[col] = ['sum', 'max']
    for col in ['Capacity', 'Manufacturer', 'Model', 'PN', 'region']:
        params[col] = 'first'
    df = df.groupby('day', as_index=False).agg(params)
    df.columns = [
        '_'.join(col) if isinstance(col, tuple) else col for col in df.columns
    ]
    df = df.sort_values('datetime',ascending=False).reset_index(drop=True)
    for col in df.columns:
        df[f'{col}_diff'] = df[col].diff()
        df[f'{col}_bdiff'] = df[col].diff(-1)
    df['ratio'] = ratio
    sn_id = 'sn_' + path.replace('.feather', '').split('_')[-1]
    df['sn_id'] = sn_id
    train = df[df['datetime'] <= pd.to_datetime('2024-06-01')].reset_index(drop=True)
    test = df[df['datetime'] > pd.to_datetime('2024-06-01')].reset_index(drop=True)
    if len(train):
        train.to_feather(f'{train_path}/{sn_id}.feather')
    if len(test):
        test.to_feather(f'{test_path}/{sn_id}.feather')

y = pd.read_csv('stage1_feather/ticket.csv')
y['datetime'] = pd.to_datetime(y['alarm_time'], unit='s')
y['left'] = y['datetime'] - pd.Timedelta(days=7)
y['right'] = y['datetime']
df = []
for path in glob.glob('stage1_feather/train/*'):
    df.append(pd.read_feather(path))
df = pd.concat(df, ignore_index=True)
df['label'] = 0
df = df.merge(y[['sn_id', 'left', 'right']], on='sn_id', how='left')
df.loc[(df['datetime'] >= df['left']) & (df['datetime'] <= df['right']), 'label'] = 1
use_cols = [x for x in df.columns if x not in ['sn_id', 'sn_type', 'datetime', 'label']]
for col in [
        'PN', 'region', 'Capacity', 'FrequencyMHz', 'Manufacturer', 'Model', 
]:
    df[col] = df[col].astype('category')
y = df['label']
df = df.drop(columns=['sn_id', 'left', 'right', 'label'])
kf = StratifiedKFold(n_splits=5)
params = {
    'objective': 'binary',
    'learning_rate': 0.1,
    'max_depth': 6,
    'early_stopping_rounds': 50
}
models = []
for fold, (train_index, valid_index) in enumerate(kf.split(df, y)):
    X_train, y_train = df.iloc[train_index][use_cols], y.iloc[train_index]
    X_valid, y_valid = df.iloc[valid_index][use_cols], y.iloc[valid_index]
    train_data = lgb.Dataset(data=X_train, label=y_train)
    valid_data = lgb.Dataset(data=X_valid, label=y_valid)
    model = lgb.train(params,
                      train_data,
                      valid_sets=[valid_data],
                      num_boost_round=2000)
    models.append(model)
df = []
for path in glob.glob('stage1_feather/test/*'):
    df.append(pd.read_feather(path))
df = pd.concat(df, ignore_index=True)
for col in [
        'PN', 'region', 'Capacity', 'FrequencyMHz', 'Manufacturer', 'Model', 
]:
    df[col] = df[col].astype('category')
ovr_preds = np.zeros((len(df), ))
for model in models:
    pred = model.predict(df[use_cols])
    ovr_preds += pred / 5
df['sub'] = ovr_preds
df = df[df['sub'] == df.groupby('sn_id')['sub'].transform('max')]
df = df[['sn_id', 'sub', 'datetime']]
df['LogTime'] = (df['datetime'] - pd.Timedelta(days=1)).view('int64') // 10**9
df['serial_number_type'] = 'A'
df = df[['sn_id', 'LogTime', 'serial_number_type']].reset_index(drop=True)
df.columns = ['sn_name', 'prediction_timestamp', 'serial_number_type']
df.to_csv(f'submission.csv', index=False, encoding='utf-8')


# In[ ]:





# In[ ]:




