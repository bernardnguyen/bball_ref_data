import pandas as pd
import numpy as np
import xgboost as xgb
import dill

# Number of years (including prediction)
N=4
# Year of prediction
YEAR = 2020

# Import data
data_df = pd.read_csv('data/test_data_%d_years_%d.csv' % (N,YEAR), index_col=0)

# Filter categories
N_CATS = int(len(data_df.columns) / (N-1))
# Cut off _Y# from categories
CATS = [c[:-3] for c in data_df.columns[:N_CATS]]
pred_df = pd.DataFrame(index=data_df.index, columns=data_df.columns[:N_CATS])
# Process output data
# Drop output columns Age, GP, MP_* and *_std
pred_df = pred_df.drop(columns = pred_df.columns[:4])
pred_df = pred_df.drop(columns=[c for c in pred_df.columns if '_avg' not in c])
pred_df = pred_df.rename(columns = {c:c[:-7] for c in pred_df.columns})
CATS = pred_df.columns

for c in CATS:
	with open('models/%s_%d.xgbm' % (c,N), 'rb') as fin:
		model = dill.load(fin)
	model_pred = model.predict(data_df.values)
	pred_df.loc[:,c] = model_pred

# Change player IDs to player names
player_summaries = pd.read_csv('data/player_summaries.csv', index_col=0)
player_summaries = player_summaries.loc[~player_summaries.index.duplicated(keep='first')]
for pid in pred_df.index:
	pred_df.loc[pid,'Name'] = player_summaries.loc[pid,'Name']

pred_df = pred_df.set_index('Name')
pred_df.to_csv('data/predictions_%d.csv' % YEAR)
