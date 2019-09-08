import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate
import dill

def main(N=3, save=True, verbose=False):
	data_df = pd.read_csv('data/training_data_%d_years.csv' % N)
	N_CATS = int(len(data_df.columns) / N)
	# Cut off _Y# from categories
	CATS = [c[:-3] for c in data_df.columns[:N_CATS]]



	# Split input and output data
	X = data_df.iloc[:,:(N_CATS*N)]
	y = data_df.iloc[:,-N_CATS:]

	# Process output data
	# Drop output columns Age, GP, MP_* and *_std
	y = y.drop(columns = y.columns[:4])
	y = y.drop(columns=[c for c in y.columns if '_avg' not in c])
	y = y.rename(columns = {c:c[:-7] for c in y.columns})

	# Split train/test data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



	# Generate XGBoost Regressor Model
	if verbose:
		print('XGBoost Regressor Results:')
	for col in y.columns:
		model = xgb.XGBRegressor(objective='reg:squarederror')
		model.fit(X_train.values, y_train[col].values)
		model_pred = model.predict(X_test.values)
		model_cv = cross_validate(model, X_train.values, y_train[col].values, cv=3, scoring=['neg_mean_squared_error','r2'], return_train_score=True)

		if verbose:
			print('\t%s:' % col)
			print('\tTraining Data Cross Validation:')
			print('\t\t%.4f\tNegative mean squared error' % np.mean(model_cv['train_neg_mean_squared_error']))
			print('\t\t %.4f\tR^2' % np.mean(model_cv['train_r2']))
			print('\tTesting Data:')
			print('\t\t%.4f\tNegative mean squared error' % -mean_squared_error(y_test[col],model_pred))
			print('\t\t %.4f\tR^2' % r2_score(y_test[col],model_pred))	
			print('\n')

		with open('models/%s_%d.xgbm' % (col,N),'wb') as fout:
			dill.dump(model,fout)

if __name__ == '__main__':
	# Default parameters
	N = 3
	save = True
	verbose = False

	# Parse parameters
	import sys
	if len(sys.argv) > 1:
		parameters = sys.argv[1:]
		for idx,p in enumerate(parameters):
			if p == '--N':
				assert ((idx+1) < len(parameters) and int(parameters[idx+1]) and int(parameters[idx+1]) in [3,4]), '--N requires an additional parameter: \
																														the number of years included in the \
																														model (3 or 4)'
				N = int(parameters[idx+1])
			elif p == '--nosave':
				save = False
			elif p == '--verbose':
				verbose = True

	# Generate models
	main(N=N,save=save,verbose=verbose)

