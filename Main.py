import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
# code to clean the data

def prep_and_engineer_data(file_path):
    # 1. Load your raw dataset into the parent DataFrame (df)
    print("Loading raw NBA dataset...")
    df = pd.read_csv(file_path)
    initial_rows = len(df)

    df = df[df['MATCH_TYPE'] == 'Regular'].copy()
    print(f"Isolated regular season. Removed {initial_rows - len(df)} non-regular records.")


    # 2. Filter: Remove players who didn't play a single game (GP == 0 or missing)
    df = df[df['GP'] > 0].copy()
    print(f"Removed {initial_rows - len(df)} records where players played 0 games.")

    # 3. Filter: Keep only modern era (2000-01 onward) and remove incomplete 2025 data
    # This addresses the massive historical shift where league-wide 3-point volume
    # spiked from ~16 attempts in 1995-96 to over 35 attempts per game by 2024.
    df = df[df['SEASON'] != '2024-25'].copy()  # Exclude incomplete 2025 season

    # Extract the starting year from the 'SEASON' string format (e.g., '2000-01' -> 2000)
    df['START_YEAR'] = df['SEASON'].apply(lambda x: int(str(x).split('-')[0]))
    df = df[df['START_YEAR'] >= 2000].copy()
    print(f"Isolated modern era (2000-2024). Remaining records: {len(df)}")

    # 4. Feature Engineering: Execute mathematical columns rounded to 1 decimal place
    print("Creating custom features relevant to All-Star selection...")

    # Per-Game Division Ops (Dividing raw volume stats by Games Played)
    df['PPG'] = (df['PTS'] / df['GP']).round(1)
    df['RPG'] = (df['REB'] / df['GP']).round(1)  # Rebounds per game
    df['APG'] = (df['AST'] / df['GP']).round(1)
    df['SPG'] = (df['STL'] / df['GP']).round(1)
    df['BPG'] = (df['BLK'] / df['GP']).round(1)
    df['TPG'] = (df['TO'] / df['GP']).round(1)  # Turnovers per game
    df['FG3APG'] = (df['FG3A'] / df['GP']).round(1)  # 3-Pointers Attempted per game
    df['FG3MPG'] = (df['FG3M'] / df['GP']).round(1)  # 3-Pointers Made per game
    df['FTMPG'] = (df['FTM'] / df['GP']).round(1)  # Free Throws Made per game
    df['FTAPG'] = (df['FTA'] / df['GP']).round(1)  # Free Throws Attempted per game
    df['MINPG'] = (df['MIN'] / df['GP']).round(1)  # Minutes per game

    # Season-Context Ops (Dividing metrics by standard 82-game season schedule)
    # Kept as raw floats to maintain precise efficiency rates for the Random Forest
    df['WIN_PCT'] = (df['W'] / 82).round(3)  # Team Win Percentage context
    df['GP_PCT'] = (df['GP'] / 82).round(3)  # Player availability rate

    # Drop temporary operational column to freeze data schema rigidity
    df = df.drop(columns=['START_YEAR'])

    print("Feature engineering complete.")

    df.to_csv('nba_table_preview.csv', index=False)
    return df


nba_awards_data=prep_and_engineer_data('CSV/seasonal_stats_with_awards.csv');


features=['FG_PCT','PPG','RPG','APG','SPG','BPG','TPG','FG3MPG','FTMPG','WIN_PCT','GP_PCT',
          'MINPG']

X=nba_awards_data[features]

y=nba_awards_data['All-Star']

# 1. Define your modern "holdout" test seasons as a list of strings
test_seasons = ['2022-23', '2023-24']

# 2. Create a True/False mask for rows that match those test seasons
is_test_season = nba_awards_data['SEASON'].isin(test_seasons)

# 3. Separate your Features (X) and Targets (y)
X_train = X[~is_test_season]  # The '~' means NOT in test seasons (all past data)
y_train = y[~is_test_season]

X_test = X[is_test_season]    # Just the modern test seasons
y_test = y[is_test_season]

rf=RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

# 1. Get the raw percentage of trees voting for All-Star (instead of just 1 or 0)
# This gives you an array of probabilities for each player
probabilities = rf.predict_proba(X_test)[:, 1]

# 2. Set a stricter threshold. A player must clear 70% confidence to be a '1'
strict_threshold = 0.70
y_pred_strict = (probabilities >= strict_threshold).astype(int)

# 3. Print your new classification report to see your precision jump up!
# print(classification_report(y_test, y_pred_strict, target_names=['Not All-Star', 'All-Star']))

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
                print("Please enter a valid number.")

    return value;

def convert_to_percent(decimal):
    return round((decimal * 100),2)

def get_input(message):
    input_string=input()

    while input_string.strip()=="":
        print(message)

        input_string=input()

    return input_string;


class Main:
    run=True

    while run:
        print("What is the name of the NBA player you wish to decide if they should"
          " be named an All-Star for the year?")

        player_name=get_input("Please enter a valid name.")

        print("For this player, I will be asking, in order, their 1.) overall field goal "
              "percentage, 2.) points per game, 3.) rebounds per game, 4.) assists per game, 5.) steals "
              "per game, 6.) blocks per game, 7.) turnovers per game, 8.) three-point field goals per game, "
              "9.) free-throws made per game, 10.) overall win percentage, 11.) games played percentage "
              "and 12.) minutes per game.")

        print("\n")

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

        player_minpg = get_numerical_value("What is this player's minutes per game?", 48)

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
            'GP_PCT': [player_gp_pct],
            'MINPG': [player_minpg]
        })

        player_stats_table_string=f"""
            FG_PCT:{round((player_fg_pct*100),1)}%,
            PPG: {player_ppg},
            RPG: {player_rpg},
            APG: {player_apg},
            SPG: {player_spg},
            BPG: {player_bpg},
            TPG: {player_tpg},
            FG3MPG: {player_fg3_mpg},
            FTMPG: {player_ft_mpg},
            WIN_PCT: {round((player_win_pct*100),1)}%,
            GP_PCT: {round((player_gp_pct*100),1)}%,
            MINPG: {player_minpg},"""

        prediction=rf.predict(new_player_stats)

        prob=rf.predict_proba(new_player_stats)

        is_outlier = outlier_model.predict(new_player_stats)[0]

        not_all_star_percent = convert_to_percent(prob[0][0])

        all_star_percent = convert_to_percent(prob[0][1])

        message=""

        # if all-star percentage is less than strict threshold of making all star then they  are not all star. so begin with
        # stating they are not all star first
        if all_star_percent< strict_threshold:
            message = (f"With these numbers for the season, the model estimates a"
                       f" {not_all_star_percent}% probability that {player_name} does not belong"
                       f" in the All-Star category and a {all_star_percent}% probability that "
                       f"{player_name} belongs in the All-Star category.")

        else:
            message = (f"With these numbers for the season, the model estimates a"
                       f" {all_star_percent}% probability that {player_name} belongs in the All-Star "
                       f"category and a {not_all_star_percent}% probability that {player_name} "
                       f"does not belong in the All-Star category.")

        print(message+"\n"+player_stats_table_string)

        run=False

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
