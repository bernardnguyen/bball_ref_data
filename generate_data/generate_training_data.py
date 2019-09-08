import numpy as np
import pandas as pd
from tqdm import tqdm

# Season summary includes:
# 	PlayerID,Year,Name,Age,GP,
# 	Avg and Std for:
#		MP,PTS,ORB,DRB,AST,STL,BLK,TOV,PF,FG,FGA,3P,3PA,FT,FTA 

# Load season summaries and index on PlayerID and Year
player_summaries = pd.read_csv('../data/player_summaries.csv')
player_summaries = player_summaries.fillna(0)
all_players = list(set(player_summaries['PlayerID']))
player_summaries = player_summaries.set_index(['PlayerID','Year'])

# Number of years to consider:
N = 3

# Initialize output df, including:
#	Age, GP, and
#	same avg and std categories as summaries for N consecutive years
#		if gap year, fill with 0s
CATS = ['Age','GP','MP_avg','MP_std','PTS_avg','PTS_std','ORB_avg','ORB_std','DRB_avg','DRB_std','AST_avg','AST_std',
		'STL_avg','STL_std','BLK_avg','BLK_std','TOV_avg','TOV_std','PF_avg','PF_std','FG_avg','FG_std','FGA_avg','FGA_std',
		'3P_avg','3P_std','3PA_avg','3PA_std','FT_avg','FT_std','FTA_avg','FTA_std']
Y_CATS = ['%s_Y%d' % (c,yr) for yr in range(N) for c in CATS]
training_data = pd.DataFrame(columns=Y_CATS)

# Build training data for each player for N consecutive years
idx = 0
for player in tqdm(all_players):
	player_seasons = player_summaries.loc[player]
	years = player_seasons.index
	# Skip player if they've played less than N years
	if len(years) < N:
		continue
	
	starting_years = range(min(years),max(years) - (N-2))
	for fy in starting_years:
		tmp_df = pd.DataFrame(index=[0],columns=Y_CATS)
		for i in range(N):
			year = fy + i
			if year not in years:
				tmp_df.loc[0,['%s_Y%d' % (c,i) for c in CATS]] = np.zeros(len(CATS))
			else:
				tmp_df.loc[0,['%s_Y%d' % (c,i) for c in CATS]] = player_seasons.loc[year,CATS].values
		training_data.loc[idx,:] = tmp_df.values
		idx += 1

training_data.to_csv('../data/training_data_%d_years.csv' % N, index=False)
