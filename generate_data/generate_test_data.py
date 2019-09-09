import pandas as pd
import numpy as np
from tqdm import tqdm

# Season summary includes:
# 	PlayerID,Year,Name,Age,GP,
# 	Avg and Std for:
#		MP,PTS,ORB,DRB,AST,STL,BLK,TOV,PF,FG,FGA,3P,3PA,FT,FTA 


# Number of years to include (including prediction):
N = 4
# Year to predict
YEAR = 2020
years_to_consider = range((YEAR-(N-1)), YEAR)

# Load season summaries and index on PlayerID and Year
player_summaries = pd.read_csv('../data/player_summaries.csv')
player_summaries = player_summaries[player_summaries['Year'].isin(years_to_consider)]
player_summaries = player_summaries.fillna(0)
all_players = list(set(player_summaries['PlayerID']))
player_summaries = player_summaries.set_index(['PlayerID','Year'])

# Initialize output df, including:
#	Age, GP, and
#	same avg and std categories as summaries for N consecutive years
#		if gap year, fill with 0s
CATS = ['Age','GP','MP_avg','MP_std','PTS_avg','PTS_std','ORB_avg','ORB_std','DRB_avg','DRB_std','AST_avg','AST_std',
		'STL_avg','STL_std','BLK_avg','BLK_std','TOV_avg','TOV_std','PF_avg','PF_std','FG_avg','FG_std','FGA_avg','FGA_std',
		'3P_avg','3P_std','3PA_avg','3PA_std','FT_avg','FT_std','FTA_avg','FTA_std']
Y_CATS = ['%s_Y%d' % (c,yr) for yr in range(N-1) for c in CATS]
test_data = pd.DataFrame(index=all_players, columns=Y_CATS)

# Build test data for each player in relevant years
for player in tqdm(all_players):
	player_seasons = player_summaries.loc[player]
	years = player_seasons.index
	
	for ytc in years_to_consider:
		tmp_df = pd.DataFrame(index=[player],columns=Y_CATS)
		for i in range(N-1):
			if ytc not in years:
				tmp_df.loc[player,['%s_Y%d' % (c,i) for c in CATS]] = np.zeros(len(CATS))
			else:
				tmp_df.loc[player,['%s_Y%d' % (c,i) for c in CATS]] = player_seasons.loc[ytc,CATS].values
		test_data.loc[player,:] = tmp_df.values

test_data.to_csv('../data/test_data_%d_years_%d.csv' % (N,YEAR))
