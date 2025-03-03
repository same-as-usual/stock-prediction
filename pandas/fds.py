import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Set style for better visualizations
plt.style.use('default')
sns.set()

# Connect to SQLite Database
conn = sqlite3.connect(r"D:\football prediction\dataset\archive\database.sqlite")

# Define league mapping
LEAGUE_MAPPING = {
    1: "England Premier League",
    2: "Spain Primera Division",
    3: "Germany 1. Bundesliga",
    4: "Italy Serie A",
    5: "France Ligue 1",
    6: "Netherlands Eredivisie",
    7: "Portugal Liga ZON Sagres",
    8: "Poland Ekstraklasa",
    9: "Belgium Jupiler League",
    10: "Switzerland Super League"
}

# Print available leagues
print("\nAvailable Leagues:")
for league_id, league_name in LEAGUE_MAPPING.items():
    print(f"{league_id}. {league_name}")

# Get user input for league
while True:
    try:
        league_choice = int(input("\nSelect league number (1-10): "))
        if league_choice in LEAGUE_MAPPING:
            break
        else:
            print("Invalid choice. Please select a number between 1 and 10.")
    except ValueError:
        print("Please enter a valid number.")

# Map league choice to actual league ID in database
league_id_mapping = {
    1: 1729,  # England
    2: 21518, # Spain
    3: 7809,  # Germany
    4: 10257, # Italy
    5: 4769,  # France
    6: 13274, # Netherlands
    7: 17642, # Portugal
    8: 15722, # Poland
    9: 19785, # Belgium
    10: 24558 # Switzerland
}

selected_league_id = league_id_mapping[league_choice]

# Get teams for selected league
team_query = f"""
SELECT DISTINCT t.team_api_id, t.team_long_name
FROM Team t
JOIN Match m ON t.team_api_id = m.home_team_api_id OR t.team_api_id = m.away_team_api_id
WHERE m.league_id = {selected_league_id}
ORDER BY t.team_long_name;
"""
teams_df = pd.read_sql_query(team_query, conn)
teams_dict = dict(zip(teams_df['team_api_id'].astype(str), teams_df['team_long_name']))

# Print teams with numbers
print(f"\nTeams in {LEAGUE_MAPPING[league_choice]}:")
for i, (team_id, team_name) in enumerate(teams_dict.items(), 1):
    print(f"{i}. {team_name} (ID: {team_id})")

# Get user input for teams
print("\nEnter team numbers from the above list:")
while True:
    try:
        home_team_num = int(input("Enter Home Team number: "))
        away_team_num = int(input("Enter Away Team number: "))
        
        if 1 <= home_team_num <= len(teams_dict) and 1 <= away_team_num <= len(teams_dict):
            home_team_id = list(teams_dict.keys())[home_team_num - 1]
            away_team_id = list(teams_dict.keys())[away_team_num - 1]
            break
        else:
            print(f"Please enter numbers between 1 and {len(teams_dict)}")
    except ValueError:
        print("Please enter valid numbers")

# Get matches for selected league
matches = pd.read_sql_query(f"SELECT * FROM Match WHERE league_id = {selected_league_id};", conn)
matches = matches.copy()

print(f"\nAnalyzing match: {teams_dict[home_team_id]} vs {teams_dict[away_team_id]}")

# ---------- EDA for Selected Teams ----------
print(f"\n=== Analysis for {teams_dict[home_team_id]} vs {teams_dict[away_team_id]} ===")

# 1. Head to Head Statistics
home_team_matches = matches[
    ((matches['home_team_api_id'] == int(home_team_id)) & (matches['away_team_api_id'] == int(away_team_id))) |
    ((matches['home_team_api_id'] == int(away_team_id)) & (matches['away_team_api_id'] == int(home_team_id)))
]

print("\nHead to Head Statistics:")
print(f"Total Matches Played: {len(home_team_matches)}")

# Calculate head to head wins
team1_wins = len(home_team_matches[
    ((home_team_matches['home_team_api_id'] == int(home_team_id)) & (home_team_matches['home_team_goal'] > home_team_matches['away_team_goal'])) |
    ((home_team_matches['away_team_api_id'] == int(home_team_id)) & (home_team_matches['away_team_goal'] > home_team_matches['home_team_goal']))
])
team2_wins = len(home_team_matches[
    ((home_team_matches['home_team_api_id'] == int(away_team_id)) & (home_team_matches['home_team_goal'] > home_team_matches['away_team_goal'])) |
    ((home_team_matches['away_team_api_id'] == int(away_team_id)) & (home_team_matches['away_team_goal'] > home_team_matches['home_team_goal']))
])
draws = len(home_team_matches[home_team_matches['home_team_goal'] == home_team_matches['away_team_goal']])

print(f"\n{teams_dict[home_team_id]} wins: {team1_wins}")
print(f"{teams_dict[away_team_id]} wins: {team2_wins}")
print(f"Draws: {draws}")

# 2. Recent Form (last 5 matches for each team)
def get_team_form(team_id):
    team_matches = matches[
        (matches['home_team_api_id'] == int(team_id)) |
        (matches['away_team_api_id'] == int(team_id))
    ].sort_values('date', ascending=False).head(5)
    
    form = []
    for _, match in team_matches.iterrows():
        if match['home_team_api_id'] == int(team_id):
            if match['home_team_goal'] > match['away_team_goal']:
                form.append('W')
            elif match['home_team_goal'] < match['away_team_goal']:
                form.append('L')
            else:
                form.append('D')
        else:
            if match['away_team_goal'] > match['home_team_goal']:
                form.append('W')
            elif match['away_team_goal'] < match['home_team_goal']:
                form.append('L')
            else:
                form.append('D')
    return form

print(f"\nRecent Form (Last 5 matches):")
print(f"{teams_dict[home_team_id]}: {' '.join(get_team_form(home_team_id))}")
print(f"{teams_dict[away_team_id]}: {' '.join(get_team_form(away_team_id))}")

# 3. Average Goals
home_team_goals = matches[matches['home_team_api_id'] == int(home_team_id)]['home_team_goal'].mean()
away_team_goals = matches[matches['away_team_api_id'] == int(away_team_id)]['away_team_goal'].mean()

print(f"\nAverage Goals:")
print(f"{teams_dict[home_team_id]} (Home): {home_team_goals:.2f}")
print(f"{teams_dict[away_team_id]} (Away): {away_team_goals:.2f}")

# 4. Visualization of Head to Head Results
plt.figure(figsize=(10, 6))
labels = [f"{teams_dict[home_team_id]} Wins", f"{teams_dict[away_team_id]} Wins", 'Draws']
sizes = [team1_wins, team2_wins, draws]
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Head to Head Results')
plt.show()

# Calculate team performance metrics
def calculate_team_metrics(team_id):
    home_matches = matches[matches['home_team_api_id'] == int(team_id)]
    away_matches = matches[matches['away_team_api_id'] == int(team_id)]
    
    home_goals_scored = home_matches['home_team_goal'].mean()
    home_goals_conceded = home_matches['away_team_goal'].mean()
    away_goals_scored = away_matches['away_team_goal'].mean()
    away_goals_conceded = away_matches['home_team_goal'].mean()
    
    home_wins = len(home_matches[home_matches['home_team_goal'] > home_matches['away_team_goal']])
    away_wins = len(away_matches[away_matches['away_team_goal'] > away_matches['home_team_goal']])
    total_matches = len(home_matches) + len(away_matches)
    
    win_rate = (home_wins + away_wins) / total_matches if total_matches > 0 else 0
    
    return {
        'goals_scored_avg': (home_goals_scored + away_goals_scored) / 2,
        'goals_conceded_avg': (home_goals_conceded + away_goals_conceded) / 2,
        'win_rate': win_rate
    }

# Create features for prediction
home_metrics = calculate_team_metrics(home_team_id)
away_metrics = calculate_team_metrics(away_team_id)

# Prepare training data with enhanced features
def prepare_match_features(row):
    home_team = str(row['home_team_api_id'])
    away_team = str(row['away_team_api_id'])
    
    home_metrics = calculate_team_metrics(home_team)
    away_metrics = calculate_team_metrics(away_team)
    
    return pd.Series({
        'home_goals_scored_avg': home_metrics['goals_scored_avg'],
        'home_goals_conceded_avg': home_metrics['goals_conceded_avg'],
        'home_win_rate': home_metrics['win_rate'],
        'away_goals_scored_avg': away_metrics['goals_scored_avg'],
        'away_goals_conceded_avg': away_metrics['goals_conceded_avg'],
        'away_win_rate': away_metrics['win_rate'],
        'home_team_encoded': int(home_team),
        'away_team_encoded': int(away_team)
    })

# Create enhanced features for all matches
print("\nPreparing enhanced features for model training...")
enhanced_features = matches.apply(prepare_match_features, axis=1)

# Create target variable
matches['match_outcome'] = matches.apply(
    lambda x: 'home_win' if x['home_team_goal'] > x['away_team_goal']
    else 'away_win' if x['home_team_goal'] < x['away_team_goal']
    else 'draw', axis=1
)

# Prepare final feature matrix
X = enhanced_features
y = matches['match_outcome']

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with better parameters
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)

# Prepare prediction data for the specific match
pred_features = pd.DataFrame([{
    'home_goals_scored_avg': home_metrics['goals_scored_avg'],
    'home_goals_conceded_avg': home_metrics['goals_conceded_avg'],
    'home_win_rate': home_metrics['win_rate'],
    'away_goals_scored_avg': away_metrics['goals_scored_avg'],
    'away_goals_conceded_avg': away_metrics['goals_conceded_avg'],
    'away_win_rate': away_metrics['win_rate'],
    'home_team_encoded': int(home_team_id),
    'away_team_encoded': int(away_team_id)
}])

# Scale prediction data
pred_features_scaled = scaler.transform(pred_features)

# Make prediction
prediction = rf_model.predict(pred_features_scaled)
probabilities = rf_model.predict_proba(pred_features_scaled)

# Make predictions on test set for accuracy calculation
y_pred = rf_model.predict(X_test_scaled)

# Print prediction results
print("\n=== Match Prediction ===")
print(f"\nMatch: {teams_dict[home_team_id]} vs {teams_dict[away_team_id]}")

# Modified prediction output
win_probabilities = dict(zip(rf_model.classes_, probabilities[0]))
team_probabilities = {
    teams_dict[home_team_id]: win_probabilities['home_win'],
    teams_dict[away_team_id]: win_probabilities['away_win'],
    'Draw': win_probabilities['draw']
}

# Sort probabilities from highest to lowest
sorted_probabilities = sorted(team_probabilities.items(), key=lambda x: x[1], reverse=True)

print("\nWin Probabilities:")
for team, prob in sorted_probabilities:
    print(f"{team}: {prob:.1%}")

# Print most likely outcome
most_likely_outcome = sorted_probabilities[0][0]
if most_likely_outcome == 'Draw':
    print(f"\nMost likely outcome: Draw between the teams")
else:
    print(f"\nMost likely winner: {most_likely_outcome}")

# Model performance metrics
y_pred = rf_model.predict(X_test_scaled)
print("\nModel Performance:")
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Train an additional model for goal prediction

# Prepare goal prediction features
goal_features = enhanced_features.copy()

# Train home goal predictor
home_goal_model = GradientBoostingRegressor(random_state=42)
home_goal_model.fit(X_train_scaled, matches['home_team_goal'][X_train.index])

# Train away goal predictor
away_goal_model = GradientBoostingRegressor(random_state=42)
away_goal_model.fit(X_train_scaled, matches['away_team_goal'][X_train.index])

# Predict goals
predicted_home_goals = round(home_goal_model.predict(pred_features_scaled)[0], 1)
predicted_away_goals = round(away_goal_model.predict(pred_features_scaled)[0], 1)

# Print prediction results
print("\n=== Match Prediction ===")
print(f"\nMatch: {teams_dict[home_team_id]} vs {teams_dict[away_team_id]}")

# Modified prediction output with goals
win_probabilities = dict(zip(rf_model.classes_, probabilities[0]))
team_probabilities = {
    teams_dict[home_team_id]: win_probabilities['home_win'],
    teams_dict[away_team_id]: win_probabilities['away_win'],
    'Draw': win_probabilities['draw']
}

# Sort probabilities from highest to lowest
sorted_probabilities = sorted(team_probabilities.items(), key=lambda x: x[1], reverse=True)

print("\nWin Probabilities:")
for team, prob in sorted_probabilities:
    print(f"{team}: {prob:.1%}")

# Print most likely outcome with score
print("\nPredicted Score:")
print(f"{teams_dict[home_team_id]} {predicted_home_goals:.0f} - {predicted_away_goals:.0f} {teams_dict[away_team_id]}")

# Print detailed prediction
most_likely_outcome = sorted_probabilities[0][0]
goal_difference = abs(predicted_home_goals - predicted_away_goals)

if most_likely_outcome == 'Draw':
    print(f"\nMost likely outcome: Draw between the teams")
else:
    if most_likely_outcome == teams_dict[home_team_id]:
        winning_goals = predicted_home_goals - predicted_away_goals
    else:
        winning_goals = predicted_away_goals - predicted_home_goals
    
    print(f"\nMost likely outcome: {most_likely_outcome} to win by {abs(winning_goals):.0f} goals")

# Additional Statistics
print("\nTeam Statistics:")
print(f"\n{teams_dict[home_team_id]}:")
print(f"Average goals scored (home): {home_metrics['goals_scored_avg']:.2f}")
print(f"Average goals conceded (home): {home_metrics['goals_conceded_avg']:.2f}")
print(f"Win rate: {home_metrics['win_rate']:.1%}")

print(f"\n{teams_dict[away_team_id]}:")
print(f"Average goals scored (away): {away_metrics['goals_scored_avg']:.2f}")
print(f"Average goals conceded (away): {away_metrics['goals_conceded_avg']:.2f}")
print(f"Win rate: {away_metrics['win_rate']:.1%}")

# Model performance metrics
y_pred = rf_model.predict(X_test_scaled)
print("\nModel Performance:")
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Create offensive and defensive comparison graphs
plt.figure(figsize=(12, 5))

# Offensive Comparison
plt.subplot(1, 2, 1)
offensive_stats = {
    teams_dict[home_team_id]: home_metrics['goals_scored_avg'],
    teams_dict[away_team_id]: away_metrics['goals_scored_avg']
}
teams = list(offensive_stats.keys())
goals = list(offensive_stats.values())

bars = plt.bar(teams, goals)
plt.title('Offensive Comparison\n(Average Goals Scored per Match)')
plt.ylabel('Goals')
plt.xticks(rotation=45)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

# Defensive Comparison (lower is better)
plt.subplot(1, 2, 2)
defensive_stats = {
    teams_dict[home_team_id]: home_metrics['goals_conceded_avg'],
    teams_dict[away_team_id]: away_metrics['goals_conceded_avg']
}
teams = list(defensive_stats.keys())
goals_conceded = list(defensive_stats.values())

bars = plt.bar(teams, goals_conceded)
plt.title('Defensive Comparison\n(Average Goals Conceded per Match)')
plt.ylabel('Goals')
plt.xticks(rotation=45)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Radar Chart for Complete Team Comparison
plt.figure(figsize=(10, 10))
categories = ['Goals Scored', 'Defense (Inv)', 'Win Rate', 'Recent Form']

# Calculate recent form as a percentage (wins in last 5 games)
def calculate_form_percentage(form_list):
    return (form_list.count('W') * 100) / len(form_list)

# Get data for both teams
home_form = get_team_form(home_team_id)
away_form = get_team_form(away_team_id)

home_stats = [
    home_metrics['goals_scored_avg'],
    1/home_metrics['goals_conceded_avg'],  # Inverse for defense (higher is better)
    home_metrics['win_rate'],
    calculate_form_percentage(home_form)/100
]

away_stats = [
    away_metrics['goals_scored_avg'],
    1/away_metrics['goals_conceded_avg'],  # Inverse for defense (higher is better)
    away_metrics['win_rate'],
    calculate_form_percentage(away_form)/100
]

# Number of variables
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)

# Close the plot by appending first value
angles = np.concatenate((angles, [angles[0]]))
home_stats = np.concatenate((home_stats, [home_stats[0]]))
away_stats = np.concatenate((away_stats, [away_stats[0]]))

# Plot
ax = plt.subplot(111, projection='polar')
ax.plot(angles, home_stats, 'o-', linewidth=2, label=teams_dict[home_team_id])
ax.fill(angles, home_stats, alpha=0.25)
ax.plot(angles, away_stats, 'o-', linewidth=2, label=teams_dict[away_team_id])
ax.fill(angles, away_stats, alpha=0.25)

# Add labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title('Team Comparison Radar Chart')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()

# Print additional analysis
print("\nTeam Comparison Analysis:")
print(f"\nOffensive Strength:")
if home_metrics['goals_scored_avg'] > away_metrics['goals_scored_avg']:
    print(f"• {teams_dict[home_team_id]} has stronger offensive record with {home_metrics['goals_scored_avg']:.2f} goals per game")
else:
    print(f"• {teams_dict[away_team_id]} has stronger offensive record with {away_metrics['goals_scored_avg']:.2f} goals per game")

print(f"\nDefensive Strength:")
if home_metrics['goals_conceded_avg'] < away_metrics['goals_conceded_avg']:
    print(f"• {teams_dict[home_team_id]} has stronger defensive record conceding {home_metrics['goals_conceded_avg']:.2f} goals per game")
else:
    print(f"• {teams_dict[away_team_id]} has stronger defensive record conceding {away_metrics['goals_conceded_avg']:.2f} goals per game")

# Calculate additional features before creating heatmap
print("\nPreparing data for heatmap analysis...")

# Calculate total goals
matches['total_goals'] = matches['home_team_goal'] + matches['away_team_goal']

# Calculate win rates for each team
def calculate_team_stats(team_id):
    home_matches = matches[matches['home_team_api_id'] == int(team_id)]
    away_matches = matches[matches['away_team_api_id'] == int(team_id)]
    
    # Calculate wins
    home_wins = len(home_matches[home_matches['home_team_goal'] > home_matches['away_team_goal']])
    away_wins = len(away_matches[away_matches['away_team_goal'] > away_matches['home_team_goal']])
    
    # Calculate total matches
    total_matches = len(home_matches) + len(away_matches)
    
    # Calculate goals
    home_goals_scored = home_matches['home_team_goal'].mean()
    away_goals_scored = away_matches['away_team_goal'].mean()
    home_goals_conceded = home_matches['away_team_goal'].mean()
    away_goals_conceded = away_matches['home_team_goal'].mean()
    
    return {
        'win_rate': (home_wins + away_wins) / total_matches if total_matches > 0 else 0,
        'goals_scored_avg': (home_goals_scored + away_goals_scored) / 2 if total_matches > 0 else 0,
        'goals_conceded_avg': (home_goals_conceded + away_goals_conceded) / 2 if total_matches > 0 else 0
    }

# Calculate stats for each team
team_stats = {}
unique_teams = set(matches['home_team_api_id'].unique()) | set(matches['away_team_api_id'].unique())
for team_id in unique_teams:
    team_stats[team_id] = calculate_team_stats(team_id)

# Add calculated features to matches DataFrame
matches['home_win_rate'] = matches['home_team_api_id'].map(lambda x: team_stats[x]['win_rate'])
matches['away_win_rate'] = matches['away_team_api_id'].map(lambda x: team_stats[x]['win_rate'])
matches['home_goals_scored_avg'] = matches['home_team_api_id'].map(lambda x: team_stats[x]['goals_scored_avg'])
matches['away_goals_scored_avg'] = matches['away_team_api_id'].map(lambda x: team_stats[x]['goals_scored_avg'])
matches['home_goals_conceded_avg'] = matches['home_team_api_id'].map(lambda x: team_stats[x]['goals_conceded_avg'])
matches['away_goals_conceded_avg'] = matches['away_team_api_id'].map(lambda x: team_stats[x]['goals_conceded_avg'])

# Now create the heatmap
print("\nCreating heatmap analysis...")

# Prepare relevant features for correlation
relevant_features = [
    'home_team_goal', 'away_team_goal', 'total_goals',
    'home_win_rate', 'away_win_rate',
    'home_goals_scored_avg', 'away_goals_scored_avg',
    'home_goals_conceded_avg', 'away_goals_conceded_avg'
]

# Create correlation matrix
correlation_matrix = matches[relevant_features].corr()

# Create heatmap with enhanced styling
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            cmap='RdYlBu',  # Red-Yellow-Blue colormap
            center=0,  # Center the colormap at 0
            fmt='.2f',  # Show 2 decimal places
            square=True,  # Make cells square
            linewidths=0.5,  # Add grid lines
            cbar_kws={"shrink": .5},  # Adjust colorbar size
            mask=np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Show only lower triangle
            )

plt.title('Feature Correlation Heatmap', pad=20, size=16)
plt.tight_layout()
plt.show()

# Additional correlation insights
print("\nStrong Correlations (|correlation| > 0.5):")
strong_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.5:
            strong_correlations.append({
                'Features': f"{correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}",
                'Correlation': correlation_matrix.iloc[i, j]
            })

# Sort and display strong correlations
strong_correlations_df = pd.DataFrame(strong_correlations)
if not strong_correlations_df.empty:
    strong_correlations_df = strong_correlations_df.sort_values('Correlation', key=abs, ascending=False)
    print(strong_correlations_df)
else:
    print("No strong correlations found (|correlation| > 0.5)")

# Additional heatmaps
plt.figure(figsize=(15, 5))

# Goals Heatmap
plt.subplot(1, 3, 1)
goals_heatmap = pd.crosstab(matches['home_team_goal'], matches['away_team_goal'])
sns.heatmap(goals_heatmap, cmap='YlOrRd', annot=True, fmt='d')
plt.title('Score Distribution Heatmap')
plt.xlabel('Away Team Goals')
plt.ylabel('Home Team Goals')

# Win Rate vs Goals Heatmap
plt.subplot(1, 3, 2)
win_rate_goals = pd.crosstab(
    pd.qcut(matches['home_win_rate'], 4, duplicates='drop'),
    pd.qcut(matches['home_team_goal'], 4, duplicates='drop')
)
sns.heatmap(win_rate_goals, cmap='viridis', annot=True, fmt='d')
plt.title('Win Rate vs Goals Relationship')
plt.xlabel('Goals Scored (Quartiles)')
plt.ylabel('Win Rate (Quartiles)')

# Form vs Goals Heatmap
plt.subplot(1, 3, 3)
recent_form_goals = pd.crosstab(
    pd.qcut(matches['home_goals_scored_avg'], 4, duplicates='drop'),
    pd.qcut(matches['away_goals_scored_avg'], 4, duplicates='drop')
)
sns.heatmap(recent_form_goals, cmap='coolwarm', annot=True, fmt='d')
plt.title('Home vs Away Scoring Form')
plt.xlabel('Away Team Scoring (Quartiles)')
plt.ylabel('Home Team Scoring (Quartiles)')

plt.tight_layout()
plt.show()

# Print insights from heatmaps
print("\nKey Insights from Heatmaps:")
print("1. Score Distribution:")
most_common_score = goals_heatmap.values.max()
print(f"• Most common scoreline appears {most_common_score} times")

print("\n2. Win Rate vs Goals:")
print("• Higher win rates generally correlate with more goals scored")

print("\n3. Team Form Analysis:")
print("• Teams' scoring patterns show interesting relationships between home and away performance")

# Close connection
conn.close()