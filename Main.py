import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

nba_awards_data=pd.read_csv('CSV/seasonal_stats_with_awards_filtered.csv');

features=['FG_PCT','PPG','RPG','APG','SPG','BPG','TPG','FG3MPG','FTMPG','WIN_PCT','GP_PCT']

X=nba_awards_data[features]

# X=nba_awards_data.drop(columns=['PTS','REB','SEASON','MATCH_TYPE','PLAYER_NAME',
#                                 'PLAYER_ID','STL','BLK','TO',
#                                 'GP','W','L','MIN','PTS', 'REB','AST', 'STL',
#                                 'BLK','TO','FGM','FGA','FG3M','FG3A','FTM','FTA',
#                                 'FGM_2','FG3M_2','FTM_2','DD','TD','FP','PIE','FG3_PCT',
#                                 'FT_PCT','MVP','ROY','DPOY','MIP','6MOY',
#                                 'All-NBA-Team','All-Defensive-Team','All-Rookie-Team',
#                                 'All-Star-MVP','Finals-MVP','POTW','POTM','ROTM',
#                                 'ROOKIE_SEASON','FG3APG','FTAPG','All-Star'
#                                 ])

y=nba_awards_data['All-Star']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

rf=RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

# for detecting unusual combinations
outlier_model = IsolationForest(
    contamination=0.01,
    random_state=42
)

outlier_model.fit(X_train)

def get_numerical_value(string, max_value):
    print(string +" Value has to be between 0 and "+str(max_value)+".")

    value = None

    while True:
        try:
            value = float(input())

            float_val = float(value)

            if float_val<0:
                raise ValueError("Value cannot be negative")

            if float_val < 0 or float_val > max_value:
                raise ValueError("Value must be between 0 and max_value")

            break;
        except ValueError as e:
            if e.args[0]=="Value cannot be negative":
                print("Value cannot be negative. Please try again.")
            elif e.args[0]=="Value must be between 0 and max_value":
                print("Please enter a valid number between 0 and "+ str(max_value))
            else:
                print("Please enter a valid number")

    return value;

def convert_to_percent(decimal):
    return round((decimal * 100),2)


class Main:
    run=True

    while run:
        print("What is the name of the NBA player you wish to decide if they should"
          " be an all star?")

        player_name=input()

        player_fg_pct=get_numerical_value("What is this player's field goal percentage in "
                                          "decimal form?",1)

        player_ppg = get_numerical_value("What is this player's points per game?", 36)

        player_fg3_mpg = get_numerical_value("What is this player's three-pointers made per "
                                             "game?", 6)

        player_ft_mpg = get_numerical_value("What is this player's free-throws made per "
                                             "game?", 11)

        player_rpg = get_numerical_value("What is this player's rebounds per game?", 16)

        player_apg = get_numerical_value("What is this player's assists per game?", 12)

        player_spg = get_numerical_value("What is this player's steals per game?", 3)

        player_bpg = get_numerical_value("What is this player's blocks per game?", 4)

        player_tpg = get_numerical_value("What is this player's turnovers per game?", 6)

        player_win_pct = get_numerical_value("What is this player's win percentage in "
                                            "decimal form?", 1)

        player_gp_pct = get_numerical_value("What is this player's games played percentage in "
                                            "decimal form?", 1)

        new_player_stats = pd.DataFrame({
            'FG_PCT': [player_fg_pct],
            'PPG': [player_ppg],
            'RPG': [player_rpg],
            'APG': [player_apg],
            'SPG': [player_spg],
            'BPG': [player_bpg],
            'TPG': [player_tpg],
            'FG3MPG': [player_fg3_mpg],
            'FTMPG': [player_ft_mpg],
            'WIN_PCT': [player_win_pct],
            'GP_PCT': [player_gp_pct]
        })

        prediction=rf.predict(new_player_stats)

        prob=rf.predict_proba(new_player_stats)

        is_outlier = outlier_model.predict(new_player_stats)[0]

        not_all_star_percent = convert_to_percent(prob[0][0])

        all_star_percent = convert_to_percent(prob[0][1])

        if prediction[0]==0:
            message = (f"{player_name} will probably not make the all-star game. There is a "
                       f"{not_all_star_percent}% chance {player_name}will not make the "
                       f"all-star game and a {all_star_percent}% chance {player_name} will "
                       f"make the all-star game.")

            print(message)
        else:
            message = (f"{player_name} should make the all-star game. There is a"
                       f" {all_star_percent}% chance {player_name} will make the all-star "
                       f"game and a {not_all_star_percent}% chance {player_name} will not "
                       f"make the all-star game.")

            print(message)

        print("\n")

        print("Do you want to continue with another player? Type y to continue"
                  " or n to stop.")

        while True:
            try:
                value=input().lower()

                if not (value== 'y' or value== 'n'):
                    raise ValueError("Value must be 'y' or 'n'")

                if value=='n':
                    run = False
                    print("Goodbye")
                else:
                    run = True

                break
            except ValueError:
                print("Please type y or n")





