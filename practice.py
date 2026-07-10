# FIRST GRAPH
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
#
#
# df=pd.read_csv('CSV/seasonal_stats_with_awards_filtered.csv');
#
# features = [
#     'PPG',
#     'FG_PCT',
#     'RPG',
#     'APG',
#     'SPG',
#     'BPG',
#     'WIN_PCT'
# ]
#
#
# # Calculate average Non-All-Star stats
# average_non_all_star = (
#     df[df['All-Star'] == 0][features]
#     .mean()
# )
#
#
# # Calculate average All-Star stats
# average_all_star = (
#     df[df['All-Star'] == 1][features]
#     .mean()
# )
#
#
# # Create MinMaxScaler
# scaler = MinMaxScaler()
#
#
# # Fit scaler using all players
# scaler.fit(df[features])
#
#
# # Normalize Non-All-Star averages
# non_all_star_scaled = scaler.transform(
#     average_non_all_star.to_frame().T
# )[0]
#
#
# # Normalize All-Star averages
# all_star_scaled = scaler.transform(
#     average_all_star.to_frame().T
# )[0]
#
#
# # Number of features
# num_features = len(features)
#
#
# # Create angles
# angles = np.linspace(
#     0,
#     2 * np.pi,
#     num_features,
#     endpoint=False
# ).tolist()
#
#
# # Save original angles before closing chart
# label_angles = angles.copy()
#
#
# # Close radar chart
# non_all_star_scaled = np.concatenate([
#     non_all_star_scaled,
#     [non_all_star_scaled[0]]
# ])
#
#
# all_star_scaled = np.concatenate([
#     all_star_scaled,
#     [all_star_scaled[0]]
# ])
#
#
# angles += angles[:1]
#
#
# # Create radar chart
# fig, ax = plt.subplots(
#     figsize=(10, 10),
#     subplot_kw={'polar': True}
# )
#
#
# # Plot Non-All-Star
# ax.plot(
#     angles,
#     non_all_star_scaled,
#     linewidth=2,
#     label='Average Non-All-Star'
# )
#
#
# ax.fill(
#     angles,
#     non_all_star_scaled,
#     alpha=0.25
# )
#
#
# # Plot All-Star
# ax.plot(
#     angles,
#     all_star_scaled,
#     linewidth=2,
#     label='Average All-Star'
# )
#
#
# ax.fill(
#     angles,
#     all_star_scaled,
#     alpha=0.25
# )
#
#
# # Feature labels
# ax.set_xticks(label_angles)
#
# ax.set_xticklabels(
#     features,
#     fontsize=11
# )
#
#
# # Set normalized scale
# ax.set_ylim(0, 1)
#
#
# # Add actual Non-All-Star values
# for angle, scaled_value, actual_value, feature in zip(
#     label_angles,
#     non_all_star_scaled[:-1],
#     average_non_all_star.values,
#     features
# ):
#
#     # Format percentages
#     if feature in ['FG_PCT', 'WIN_PCT']:
#         label = f'{actual_value:.1%}'
#     else:
#         label = f'{actual_value:.1f}'
#
#     ax.text(
#         angle,
#         scaled_value + 0.04,
#         label,
#         ha='center',
#         va='center',
#         fontsize=9
#     )
#
#
# # Add actual All-Star values
# for angle, scaled_value, actual_value, feature in zip(
#     label_angles,
#     all_star_scaled[:-1],
#     average_all_star.values,
#     features
# ):
#
#     # Format percentages
#     if feature in ['FG_PCT', 'WIN_PCT']:
#         label = f'{actual_value:.1%}'
#     else:
#         label = f'{actual_value:.1f}'
#
#     ax.text(
#         angle,
#         scaled_value + 0.04,
#         label,
#         ha='center',
#         va='center',
#         fontsize=9,
#         fontweight='bold'
#     )
#
#
# # Title
# plt.title(
#     'Average Non-All-Star vs Average All-Star',
#     size=16,
#     pad=20
# )
#
#
# # Legend
# plt.legend(
#     loc='upper right',
#     bbox_to_anchor=(1.3, 1.1)
# )
#
#
# plt.show()

#########################################################################

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
#
# nba_awards_data=pd.read_csv('CSV/seasonal_stats_with_awards_filtered.csv');
#
# features=['FG_PCT','PPG','RPG','APG','SPG','BPG','TPG','FG3MPG','FTMPG','WIN_PCT','GP_PCT']
#
# X=nba_awards_data[features]
#
# y=nba_awards_data['All-Star']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                     stratify=y, random_state=42)
#
# rf=RandomForestClassifier(
#     n_estimators=200,
#     max_depth=10,
#     random_state=42,
#     class_weight='balanced'
# )
#
# rf.fit(X_train, y_train)
#
# # Create feature importance DataFrame
# feature_importance = pd.DataFrame({
#     'Feature': features,
#     'Importance': rf.feature_importances_
# })
#
#
# # Sort features from highest to lowest importance
# feature_importance = feature_importance.sort_values(
#     by='Importance',
#     ascending=True
# )
#
#
# # Create bar chart
# plt.figure(figsize=(10, 6))
#
# plt.barh(
#     feature_importance['Feature'],
#     feature_importance['Importance']
# )
#
#
# # Add importance values to each bar
# for index, value in enumerate(feature_importance['Importance']):
#     plt.text(
#         value,
#         index,
#         f'{value:.3f}',
#         va='center'
#     )
#
#
# # Chart title
# plt.title(
#     'Random Forest Feature Importance for All-Star Prediction',
#     fontsize=16
# )
#
#
# # Axis labels
# plt.xlabel('Feature Importance')
#
# plt.ylabel('Player Statistic')
#
#
# # Adjust layout
# plt.tight_layout()
#
#
# # Display chart
# plt.show()

#########################################################################



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

nba_awards_data=pd.read_csv('CSV/seasonal_stats_with_awards_filtered.csv');

features=['FG_PCT','PPG','RPG','APG','SPG','BPG','TPG','FG3MPG','FTMPG','WIN_PCT','GP_PCT']

X=nba_awards_data[features]

y=nba_awards_data['All-Star']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)

# 2. Set up a large plot canvas so the text doesn't overlap
plt.figure(figsize=(20, 10))

# 3. Pull ONE tree out of the forest and plot a simplified version
# We use max_depth=3 here strictly for the visual layout,
# ensuring it stays clean and readable for the user.
plot_tree(
    rf_model.estimators_[0],          # Pulls the first tree out of your 100 trees
    max_depth=3,                      # Limits visual depth so it doesn't get cluttered
    feature_names=features,     # Uses your list of 10-12 NBA stat labels
    class_names=['Role Player', 'All-Star'], # Maps 0 and 1 to clean sports terms
    filled=True,                      # Colors the boxes (Blue for All-Stars, Orange for Role Players)
    rounded=True,                     # Rounds the box corners for a cleaner modern UI vibe
    fontsize=10                       # Adjust font size for scannability
)

# 4. Save and show the flowchart
plt.title("Inside an NBA All-Star Decision Tree", fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("simplified_nba_decision_tree.png", bbox_inches='tight')
plt.show()
