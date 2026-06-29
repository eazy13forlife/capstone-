import pandas as pd
from sklearn.tree import DecisionTreeClassifier

nba_awards_data=pd.read_csv('CSV/seasonal_stats_with_awards_filtered.csv');

X=nba_awards_data.drop(columns=['SEASON','MATCH_TYPE','PLAYER_NAME', 'PLAYER_ID',
                                'GP','W','L','MIN','PTS', 'REB','AST', 'STL',
                                'BLK','TO','FGM','FGA','FG3M','FG3A','FTM','FTA',
                                'FGM_2','FG3M_2','FTM_2','DD','TD','FP','PIE',
                                'FT_PCT','MVP','ROY','DPOY','MIP','6MOY',
                                'All-NBA-Team','All-Defensive-Team','All-Rookie-Team',
                                'All-Stat_MVP','Finals-MVP','POTW','POTM','ROTM',
                                'ROOKIE_SEASON','FG3APG','FG3MPG','FTMPG','FTAPG'
                                ])

y=nba_awards_data['All-Star']

print(y)