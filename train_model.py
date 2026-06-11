import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# =========================
# LOAD DATASETS
# =========================

matches = pd.read_csv('data/matches.csv')
deliveries = pd.read_csv('data/deliveries.csv')

# =========================
# MERGE DATASETS
# =========================

total_df = deliveries.merge(
    matches[['id', 'season']],
    left_on='match_id',
    right_on='id'
)

# =========================
# IPL 2026 TEAMS
# =========================

teams = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad'
]

# =========================
# TEAM NAME REPLACEMENTS
# =========================

team_replacements = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
}

total_df['batting_team'] = total_df[
    'batting_team'
].replace(team_replacements)

total_df['bowling_team'] = total_df[
    'bowling_team'
].replace(team_replacements)

# =========================
# FILTER TEAMS
# =========================

total_df = total_df[
    total_df['batting_team'].isin(teams)
]

total_df = total_df[
    total_df['bowling_team'].isin(teams)
]

# =========================
# TOTAL RUNS
# =========================

total_df['total_runs'] = (
    total_df['batsman_runs'] +
    total_df['extra_runs']
)

# =========================
# CURRENT SCORE
# =========================

total_df['current_score'] = total_df.groupby(
    'match_id'
)['total_runs'].cumsum()

# =========================
# BALLS BOWLED
# =========================

total_df['balls_bowled'] = (
    (total_df['over'] - 1) * 6 +
    total_df['ball']
)

# Remove invalid rows
total_df = total_df[
    total_df['balls_bowled'] > 0
]

total_df = total_df[
    total_df['balls_bowled'] <= 120
]

# =========================
# BALLS LEFT
# =========================

total_df['balls_left'] = (
    120 - total_df['balls_bowled']
)

# =========================
# WICKETS
# =========================

total_df['player_dismissed'] = total_df[
    'player_dismissed'
].fillna("0")

total_df['wickets'] = total_df[
    'player_dismissed'
].apply(
    lambda x: 0 if x == "0" else 1
)

total_df['wickets'] = total_df.groupby(
    'match_id'
)['wickets'].cumsum()

# =========================
# WICKETS LEFT
# =========================

total_df['wickets_left'] = (
    10 - total_df['wickets']
)

# =========================
# CURRENT RUN RATE
# =========================

total_df['crr'] = (
    total_df['current_score'] * 6 /
    total_df['balls_bowled']
)

# =========================
# FINAL SCORE
# =========================

final_scores = total_df.groupby(
    'match_id'
)['current_score'].max().reset_index()

final_scores.rename(
    columns={'current_score': 'final_score'},
    inplace=True
)

total_df = total_df.merge(
    final_scores,
    on='match_id'
)

# =========================
# REALISTIC FILTERS
# =========================

total_df = total_df[
    total_df['current_score'] > 20
]

total_df = total_df[
    total_df['final_score'] < 280
]

total_df = total_df[
    total_df['balls_left'] > 0
]

# =========================
# FINAL DATAFRAME
# =========================

final_df = total_df[[
    'batting_team',
    'bowling_team',
    'current_score',
    'balls_left',
    'wickets_left',
    'crr',
    'final_score'
]]

# Remove missing/infinite values
final_df = final_df.dropna()

# =========================
# FEATURES & TARGET
# =========================

X = final_df.drop(columns=['final_score'])
y = final_df['final_score']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# ENCODING
# =========================

trf = ColumnTransformer([
    (
        'trf',
        OneHotEncoder(
            sparse_output=False,
            drop='first'
        ),
        ['batting_team', 'bowling_team']
    )
], remainder='passthrough')

# =========================
# MODEL PIPELINE
# =========================

pipe = Pipeline([
    ('step1', trf),

    ('step2', RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

# =========================
# TRAIN MODEL
# =========================

pipe.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = pipe.predict(X_test)

# =========================
# MODEL ACCURACY
# =========================

score = r2_score(y_test, y_pred)

print("\nModel Accuracy:", round(score, 2))

# =========================
# SAVE MODEL
# =========================

joblib.dump(pipe, 'models/model.pkl')

print("\nRealistic IPL Model Saved Successfully!")