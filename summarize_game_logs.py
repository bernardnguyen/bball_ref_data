from PandasBasketball import pandasbasketball as pb
import pandas as pd
from tqdm import tqdm

# Category choices:
#    G, Date, Age, Tm,  , Opp,  , GS, MP, FG, FGA, FG%, 3P, 3PA, 3P%, FT, 
#    FTA, FT%, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, GmSc
CATS = ['MP','FG','FGA','3P','3PA','FT','FTA','ORB','DRB','AST','STL','BLK','TOV','PF','PTS']
player_list = pd.read_csv('data/players_failed_to_load.csv', header=0)

CATS_AVG = ['%s_avg' % c for c in CATS]
CATS_STD = ['%s_std' % c for c in CATS]
CATS_SUM = ['Name','PlayerID','Year'] + CATS_AVG + CATS_STD

player_summaries = pd.DataFrame(columns=CATS_SUM)
failed = pd.DataFrame(columns=player_list.columns)


for i,p in tqdm(player_list.iterrows()):
	[player_id, name, year] = p
	try:
		gamelogs = pb.get_player_gamelog(player_id, year)
		gamelogs = gamelogs[CATS]
		gamelogs['MP'] = [mp.split(':')[0] for mp in gamelogs['MP']]

		averages = list(gamelogs.astype('int64').mean().values)
		stdevs = list(gamelogs.astype('int64').std().values)
		player_summaries.loc[i] = [name,player_id,year] + averages + stdevs
	except:
		try:
			gamelogs = pb.get_player_gamelog(player_id, year)
			gamelogs = gamelogs[CATS]
			gamelogs['MP'] = [mp.split(':')[0] for mp in gamelogs['MP']]

			averages = list(gamelogs.astype('int64').mean().values)
			stdevs = list(gamelogs.astype('int64').std().values)
			player_summaries.loc[i] = [name,player_id,year] + averages + stdevs
		except:
			try:
				gamelogs = pb.get_player_gamelog(player_id, year)
				gamelogs = gamelogs[CATS]
				gamelogs['MP'] = [mp.split(':')[0] for mp in gamelogs['MP']]

				averages = list(gamelogs.astype('int64').mean().values)
				stdevs = list(gamelogs.astype('int64').std().values)
				player_summaries.loc[i] = [name,player_id,year] + averages + stdevs
			except:
				failed = failed.append({'PlayerID':player_id,'Name':name,'Year':year},ignore_index=True)
				pass

print(len(failed))
player_summaries.to_csv('data/player_summaries.csv',mode='a',index=False)
failed.to_csv('data/players_failed_to_load.csv',index=False)