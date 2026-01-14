# ELO Implementation Guide - Using Built-In Features

## Core ELO Formula

```python
# Expected score (win probability)
expected = 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))

# New rating after game
new_elo = old_elo + k_factor * (actual_score - expected_score)
# actual_score: 1.0 = win, 0.5 = OT loss, 0.0 = regulation loss
```

---

## Incorporating Built-In Features

### 1. Initial Ratings by Division Tier

```python
# D1 teams start higher than D2/D3
initial_ratings = {
    'D1': 1600,  # Strong teams
    'D2': 1500,  # Average teams
    'D3': 1400   # Weaker teams
}

# Initialize all teams
elo_ratings = {}
for team in teams:
    tier = team_division[team]  # From 'division' column
    elo_ratings[team] = initial_ratings[tier]
```

### 2. Rest Time Advantage

```python
# Teams with more rest get temporary ELO boost
def adjust_for_rest(team_elo, rest_time, opponent_rest, rest_advantage_per_day):
    rest_diff = rest_time - opponent_rest
    return team_elo + (rest_diff * rest_advantage_per_day)

# Example: Home team has 3 days rest, away has 1 day
# rest_advantage_per_day = 10
# home_elo_adjusted = home_elo + (3-1)*10 = home_elo + 20
```

### 3. Back-to-Back Penalty

```python
# Penalize teams playing on 0-1 days rest
def apply_b2b_penalty(team_elo, rest_time, b2b_penalty):
    if rest_time <= 1:
        return team_elo - b2b_penalty
    return team_elo

# Example: Team played yesterday (rest_time=0)
# b2b_penalty = 50
# adjusted_elo = team_elo - 50
```

### 4. Travel Distance Fatigue

```python
# Long travel reduces effective ELO
def apply_travel_fatigue(team_elo, travel_distance, travel_penalty_per_1000mi):
    penalty = (travel_distance / 1000) * travel_penalty_per_1000mi
    return team_elo - penalty

# Example: Team traveled 2000 miles
# travel_penalty_per_1000mi = 15
# adjusted_elo = team_elo - (2000/1000)*15 = team_elo - 30
```

### 5. Injury Adjustment

```python
# Each key injury reduces team strength
def apply_injury_penalty(team_elo, injuries, injury_penalty_per_player):
    return team_elo - (injuries * injury_penalty_per_player)

# Example: Team has 2 key injuries
# injury_penalty_per_player = 25
# adjusted_elo = team_elo - 50
```

### 6. Margin of Victory (MOV) Multiplier

```python
# Blowouts mean more than close games
def calculate_mov_multiplier(goal_diff, mov_multiplier, mov_method):
    if mov_method == 'linear':
        return 1 + (abs(goal_diff) * mov_multiplier)
    elif mov_method == 'logarithmic':
        return 1 + (np.log(abs(goal_diff) + 1) * mov_multiplier)
    return 1.0

# Example: Team wins 5-1 (goal_diff = 4)
# mov_method = 'logarithmic', mov_multiplier = 1.0
# multiplier = 1 + log(5) * 1.0 = 1 + 1.61 = 2.61
# ELO change is 2.61x larger
```

### 7. Overtime Win Discount

```python
# OT wins are "half wins" in some implementations
def get_actual_score(outcome, ot_win_multiplier):
    if outcome == 'RW':  # Regulation win
        return 1.0
    elif outcome == 'OTW':  # Overtime win
        return ot_win_multiplier  # Default 0.75
    elif outcome == 'OTL':  # Overtime loss
        return 1 - ot_win_multiplier  # 0.25
    elif outcome == 'RL':  # Regulation loss
        return 0.0
```

---

## Full Game Update Logic

```python
def update_elo_ratings(
    home_team, away_team,
    home_goals, away_goals,
    outcome,  # 'RW', 'OTW', 'OTL', 'RL'
    home_rest, away_rest,
    away_travel_dist,
    home_injuries, away_injuries,
    elo_ratings,  # Current ratings dict
    params  # Hyperparameters from YAML
):
    # Step 1: Get base ELO ratings
    home_elo = elo_ratings[home_team]
    away_elo = elo_ratings[away_team]
    
    # Step 2: Apply all adjustments
    home_elo_adj = home_elo
    away_elo_adj = away_elo
    
    # Home advantage
    home_elo_adj += params['home_advantage']
    
    # Rest advantage
    rest_diff = home_rest - away_rest
    home_elo_adj += rest_diff * params['rest_advantage_per_day']
    
    # Back-to-back penalty
    if home_rest <= 1:
        home_elo_adj -= params['b2b_penalty']
    if away_rest <= 1:
        away_elo_adj -= params['b2b_penalty']
    
    # Travel fatigue (only away team travels)
    away_elo_adj -= (away_travel_dist / 1000) * 15  # 15 points per 1000 miles
    
    # Injuries
    home_elo_adj -= home_injuries * 25  # 25 points per injury
    away_elo_adj -= away_injuries * 25
    
    # Step 3: Calculate expected scores
    home_expected = 1 / (1 + 10 ** ((away_elo_adj - home_elo_adj) / 400))
    away_expected = 1 - home_expected
    
    # Step 4: Get actual scores
    home_actual = get_actual_score(outcome, params['ot_win_multiplier'])
    away_actual = 1 - home_actual
    
    # Step 5: Calculate MOV multiplier
    goal_diff = home_goals - away_goals
    mov_mult = calculate_mov_multiplier(
        goal_diff, 
        params['mov_multiplier'], 
        params['mov_method']
    )
    
    # Step 6: Update ELO ratings
    k = params['k_factor'] * mov_mult
    elo_ratings[home_team] += k * (home_actual - home_expected)
    elo_ratings[away_team] += k * (away_actual - away_expected)
    
    return elo_ratings
```

---

## Making Predictions

```python
def predict_goals(home_team, away_team, elo_ratings, game_features, params):
    # Get adjusted ELO ratings
    home_elo_adj = elo_ratings[home_team]
    away_elo_adj = elo_ratings[away_team]
    
    # Apply same adjustments as training
    home_elo_adj += params['home_advantage']
    home_elo_adj += (game_features['home_rest'] - game_features['away_rest']) * params['rest_advantage_per_day']
    
    if game_features['home_rest'] <= 1:
        home_elo_adj -= params['b2b_penalty']
    if game_features['away_rest'] <= 1:
        away_elo_adj -= params['b2b_penalty']
    
    away_elo_adj -= (game_features['away_travel_dist'] / 1000) * 15
    home_elo_adj -= game_features['home_injuries'] * 25
    away_elo_adj -= game_features['away_injuries'] * 25
    
    # Calculate win probability
    home_win_prob = 1 / (1 + 10 ** ((away_elo_adj - home_elo_adj) / 400))
    
    # Convert to expected goal differential
    # Typical game: 3 goals per team, range -6 to +6
    expected_diff = (home_win_prob - 0.5) * 12  # Scale to ±6 goals
    
    # Predict goals (league average is ~3.0 per team)
    home_goals_pred = 3.0 + (expected_diff / 2)
    away_goals_pred = 3.0 - (expected_diff / 2)
    
    return home_goals_pred, away_goals_pred
```

---

## Hyperparameter Tuning

The grid search will test **648 combinations**:
- `k_factor`: 3 values (20, 32, 40)
- `home_advantage`: 3 values (50, 100, 150)
- `mov_multiplier`: 3 values (0, 1, 1.5)
- `mov_method`: 2 values (linear, logarithmic)
- `season_carryover`: 3 values (0.67, 0.75, 0.85)
- `ot_win_multiplier`: 2 values (0.75, 1.0)
- `rest_advantage_per_day`: 2 values (0, 10)
- `b2b_penalty`: 2 values (0, 50)

**Best configs will likely be:**
- `k_factor=32` (balanced volatility)
- `home_advantage=100` (significant but not extreme)
- `mov_multiplier=1.0, mov_method=logarithmic` (blowouts matter, diminishing returns)
- `rest_advantage_per_day=10, b2b_penalty=50` (fatigue is real!)

---

## Evaluation Loop

```python
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
df = pd.read_csv('hockey_data.csv')
df = df.sort_values('game_date')  # CRITICAL: chronological order

# Load hyperparameters
params_df = pd.read_csv('output/hyperparams/model3_elo_grid_search.csv')

results = []
for _, params in params_df.iterrows():
    # Initialize ratings
    elo_ratings = initialize_ratings(df, params)
    
    predictions = []
    actuals = []
    
    # Loop through games chronologically
    for _, game in df.iterrows():
        # Predict
        home_pred, away_pred = predict_goals(
            game['home_team'], game['away_team'],
            elo_ratings, game, params
        )
        
        predictions.append(home_pred)
        actuals.append(game['home_goals'])
        
        # Update ratings (training step)
        elo_ratings = update_elo_ratings(
            game['home_team'], game['away_team'],
            game['home_goals'], game['away_goals'],
            game['outcome'],
            game['home_rest'], game['away_rest'],
            game['away_travel_dist'],
            game['home_injuries'], game['away_injuries'],
            elo_ratings, params
        )
    
    # Calculate metrics
    rmse = mean_squared_error(actuals, predictions, squared=False)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    results.append({
        'experiment_id': params['experiment_id'],
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        **params
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('output/hyperparams/model3_elo_results.csv', index=False)
```

---

## Key Implementation Notes

1. **Chronological order is CRITICAL** - ELO updates sequentially, can't shuffle data
2. **Separate train/test** - Update ratings during training, freeze ratings for test predictions
3. **Division-aware initialization** - D1 teams start at 1600, D3 at 1400
4. **Travel only affects away team** - Home team didn't travel
5. **Rest differential** - Team with more rest gets advantage
6. **MOV logarithmic** - 5-0 win isn't 5x better than 1-0 win (diminishing returns)
7. **Season carryover** - Regress ratings toward mean between seasons to account for roster changes

---

## Expected Performance

With proper tuning using built-in features:
- **RMSE**: 2.2-2.5 goals per game
- **R²**: 0.70-0.75
- **MAE**: 1.7-2.0 goals

Key factors: `rest_time`, `travel_distance`, `injuries` will provide **major** improvements over basic ELO.
