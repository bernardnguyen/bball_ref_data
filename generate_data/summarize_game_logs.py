from PandasBasketball import pandasbasketball as pb
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Category choices:
#    G, Date, Age, Tm,  , Opp,  , GS, MP, FG, FGA, FG%, 3P, 3PA, 3P%, FT, 
#    FTA, FT%, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, GmSc
CATS = ['Age','MP','PTS','ORB','DRB','AST','STL','BLK','TOV','PF','FG','FGA','3P','3PA','FT','FTA']
try:
	player_list = pd.read_csv('../data/players_failed_to_load.csv')
except:
	player_list = pd.read_csv('../data/player_list.csv')
CATS_SUM = [['%s_avg' % c, '%s_std' % c] for c in CATS[1:]]
CATS_SUM = np.array(CATS_SUM)
CATS_SUM = CATS_SUM.flatten()
CATS_SUM = ['Name','PlayerID','Year','Age','GP'] + list(CATS_SUM)

player_summaries = pd.DataFrame(columns=CATS_SUM)
failed = pd.DataFrame(columns=player_list.columns)

for idx in tqdm(player_list.index):
	[player_id, name, year] = player_list.loc[idx]
	try:
		gamelogs = pb.get_player_gamelog(player_id, year)
		gamelogs = gamelogs[CATS]
		gamelogs['Age'] = [a.split('-')[0] for a in gamelogs['Age']]
		age = gamelogs['Age'].astype('int64').mean()
		gamelogs = gamelogs.drop(columns=['Age'])

		gamelogs['MP'] = [mp.split(':')[0] for mp in gamelogs['MP']]
		gamelogs = gamelogs[gamelogs['MP'] != '']
		gamelogs = gamelogs[gamelogs['MP'] != '00']
		gamelogs = gamelogs[gamelogs['MP'] != None]

		player_summaries.loc[idx,['PlayerID','Year','Name','Age','GP']] = [player_id,year,name,age,len(gamelogs)]
		player_summaries.loc[idx,['%s_avg' % c for c in CATS[1:]]] = list(gamelogs.astype('int64').mean().values)
		player_summaries.loc[idx,['%s_std' % c for c in CATS[1:]]] = list(gamelogs.astype('int64').std().values)
	except AttributeError:
		pass
	except KeyboardInterrupt:
		raise
	except Exception as e:
		# print(e)
		if idx in player_summaries.index:
			player_summaries = player_summaries.drop(index=idx)
		failed = failed.append({'PlayerID':player_id,'Name':name,'Year':year},ignore_index=True)
		pass

print('Failed retrievals:\t%d\nSuccessful retrievals:\t%d' % (len(failed), len(player_summaries)))
if os.path.isfile('../data/player_summaries.csv'):
	player_summaries.to_csv('../data/player_summaries.csv',mode='a',index=False,header=False)
else:
	player_summaries.to_csv('../data/player_summaries.csv',index=False)
failed.to_csv('../data/players_failed_to_load.csv',index=False)