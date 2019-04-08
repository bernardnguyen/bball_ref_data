import os,glob
import pandas as pd

csv_files = glob.glob('data/Season_Totals/*.csv')

player_df = pd.DataFrame(columns=['PlayerID','Name','Year'])

for f in csv_files:
	year = f.split('/')[2][:-4]
	player_dict = {}

	df = pd.read_csv(f)
	player_list = list(set(df['Player']))
	for p in player_list:
		[player, player_id] = p.split('\\')
		
		if player_dict.get(player_id, False):
			continue
		else:
			player_dict[player_id] = True
			player_df = player_df.append({'PlayerID':player_id,
											'Name':player,
											'Year':year}, ignore_index=True)

player_df.to_csv('data/player_list.csv',index=False)