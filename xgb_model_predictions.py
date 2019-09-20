import pandas as pd
import numpy as np
import xgboost as xgb
import dill
from sklearn.preprocessing import normalize

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

# Calculate predictions
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

# Calculate impact metrics for league-specific categories
volume_stats = ['3P','PTS','REB','AST','STL','BLK']

pred_df['REB'] = pred_df['ORB'] + pred_df['DRB']
pred_df['ATR'] = pred_df['AST'] / pred_df['TOV']
pred_df['FGP'] = pred_df['FG'] / pred_df['FGA']
pred_df['FTP'] = pred_df['FT'] / pred_df['FTA']
pred_df['3PP'] = pred_df['3P'] / pred_df['3PA']

def relative_value(df, cat, nonnegative=False):
	output = ((df[cat] - df[cat].mean()) / df[cat].std())
	if nonnegative:
		output[output < 0] = 0
	return output

for c in volume_stats:
	impact = relative_value(pred_df,c, nonnegative=True)
	pred_df['%s_impact' % c] = impact

pred_df['ATR_impact'] = relative_value(pred_df,'AST') - relative_value(pred_df,'TOV')
pred_df['FGP_impact'] = relative_value(pred_df,'FGP') * relative_value(pred_df,'FGA',nonnegative=True)
pred_df['FTP_impact'] = relative_value(pred_df,'FTP') * relative_value(pred_df,'FTA',nonnegative=True)
pred_df['3PP_impact'] = relative_value(pred_df,'3PP') * relative_value(pred_df,'3PA',nonnegative=True)
impact_cols = [c for c in pred_df.columns if '_impact' in c]
pred_df[impact_cols] = normalize(pred_df[impact_cols],norm='max',axis=0)
pred_df['REL_impact'] = pred_df[impact_cols].sum(axis=1)
pred_df = pred_df.sort_values(by='REL_impact',ascending=False)

# Save the predictions
pred_df.to_csv('data/predictions_%d.csv' % YEAR)
