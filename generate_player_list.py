import os,glob
import pandas as pd

csv_files = glob.glob('data/Season_Totals/*.csv')

years_to_skip = []
try:
	players_already_had = pd.read_csv('data/player_list.csv')
	years_to_skip = list(set(players_already_had['Year']))
except:
	pass

player_df = pd.DataFrame(columns=['PlayerID','Name','Year'])

for f in csv_files:	
	year = int(f.split('/')[2][:-4])
	if year in years_to_skip:
		continue

	player_dict = {}
	df = pd.read_csv(f)
	player_list = list(set(df['Player']))
	for p in player_list:
		[player, player_id] = p.split('\\')
		
		# Only get player name/ID once per season
		if player_dict.get(player_id, False):
			continue
		else:
			player_dict[player_id] = True
			player_df = player_df.append({'PlayerID':player_id,
											'Name':player,
											'Year':year}, ignore_index=True)

# If expanding, failed to load already exists. Append to that list.
if os.path.isfile('data/players_failed_to_load.csv'):
	player_df.to_csv('data/players_failed_to_load.csv',mode='a',index=False,header=False)
# No matter the case, append to the player list file.
player_df.to_csv('data/player_list.csv',mode='a',index=False,header=False)