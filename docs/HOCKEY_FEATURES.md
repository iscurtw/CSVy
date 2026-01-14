# Hockey Data Feature Guide

## Available Features from Dataset

### Numerical Features (Built-In):
- **GP**: Games Started
- **RW**: Regulation Wins (2 points)
- **OTW**: Overtime Wins (2 points)
- **RL**: Regulation Losses (0 points)
- **OTL**: Overtime Losses (1 point)
- **PTS**: Total Points
- **P%**: Points Percentage
- **GF**: Goals For
- **GA**: Goals Against
- **DIFF**: Goal Differential (GF - GA)
- **HOME**: Home Record
- **AWAY**: Away Record
- **OT**: OT Record
- **L10**: Recent Record (Last 10 Games)
- **STRK**: Win Streak
- **travel_time**: Travel Time (hours)
- **travel_distance**: Travel Distance (miles)
- **rest_time**: Days Since Last Game
- **injuries**: Number of Key Injuries

### Categorical Features (Built-In):
- **clinch_status**: Playoff/Division/Conference clinched
- **award**: Presidents' Trophy winner (best regular season record)
- **elimination**: Playoff elimination status (eliminator/eliminated)
- **travel**: Away team traveling to home court
- **division**: D1, D2, or D3 tier

### Features to Engineer with CSVy:

#### 1. Rate Statistics (Normalize by Games Played)
```bash
ruby cli.rb rate data.csv GF GP -n goals_per_game
ruby cli.rb rate data.csv GA GP -n goals_against_per_game
ruby cli.rb rate data.csv W GP -n win_rate
ruby cli.rb rate data.csv RW GP -n regulation_win_rate
```

**Why**: Compare teams with different GP counts fairly

#### 2. Rolling/Moving Averages (Momentum)
```bash
ruby cli.rb rolling data.csv GF -w 5 -g team_name    # Last 5 games offense
ruby cli.rb rolling data.csv GA -w 5 -g team_name    # Last 5 games defense
ruby cli.rb rolling data.csv DIFF -w 10 -g team_name # Last 10 games form
ruby cli.rb rolling data.csv PTS -w 10 -g team_name  # Points trend
```

**Why**: Capture hot/cold streaks, recent form matters more than season averages

#### 3. EWMA (Exponentially Weighted Moving Average)
```bash
ruby cli.rb ewma data.csv DIFF -s 10 -g team_name  # Recent games weighted more
ruby cli.rb ewma data.csv W -s 5 -g team_name
```

**Why**: Recent games matter more - injuries, trades, momentum

#### 4. Lag Features (Previous Games)
```bash
ruby cli.rb lag data.csv W 1,3,5 -g team_name      # Wins from 1,3,5 games ago
ruby cli.rb lag data.csv DIFF 1,3,5 -g team_name   # Goal diff from prev games
ruby cli.rb lag data.csv PTS 1 -g team_name        # Points from last game
```

**Why**: Time-series models need historical context

#### 5. Home/Away Splits
Parse HOME and AWAY strings to extract:
- Home wins, losses, OT
- Away wins, losses, OT
- Home win rate vs away win rate

**Why**: Home advantage is huge in hockey

#### 6. Streak Analysis
```bash
ruby cli.rb streak data.csv result -g team_name
```

Parse STRK field: "W5" = 5 game win streak, "L3" = 3 game losing streak

**Why**: Momentum indicators

#### 7. Strength of Schedule
Calculate:
- Average opponent PTS
- Average opponent DIFF
- Number of games vs playoff teams

**Why**: Wins vs good teams > wins vs bad teams

#### 8. Playoff Pressure Indicators
- Distance from playoff line (PTS difference from 8th place)
- Clinched status (X, Y, P, Z flags)
- Games remaining

**Why**: Desperation/motivation factor

#### 9. Special Situations
From S/O record:
- Shootout win rate
- Close game performance

**Why**: Indicates clutch performance

#### 10. Normalized Features for Models
```bash
ruby cli.rb normalize data.csv GF -m minmax     # Scale 0-1
ruby cli.rb normalize data.csv GA -m zscore     # Standardize
```

**Why**: Linear regression and neural networks need scaled features

## Features for Each Model:

### Model 1 (Baseline - Simple Average):
- DIFF (goal differential)
- P% (points percentage)

### Model 2 (Linear Regression):
- All rate stats (per-game metrics)
- Home/Away splits
- One-hot encoded division/conference
- Normalized GF, GA, PTS

### Model 3 (ELO):
- W/L/OT outcomes
- DIFF (for margin-of-victory adjustment)
- HOME flag (for home advantage)
- Date/order (for sequential updates)

### Model 4 (XGBoost/Random Forest):
- Everything from Model 2
- Rolling averages (5, 10 games)
- EWMA features
- Lag features
- Streak indicators
- Strength of schedule
- Playoff pressure indicators

### Model 5 (Ensemble):
- Predictions from Models 1-4
- Confidence intervals
- Model agreement metrics

## When Data Drops - Action Plan:

1. **Immediate (Day 1)**:
   ```bash
   ruby cli.rb diagnose raw_data.csv
   ruby cli.rb clean raw_data.csv
   ruby cli.rb profile raw_data.csv
   ```

2. **Feature Engineering (Day 1-2)**:
   - Run all rate calculations
   - Generate rolling features
   - Create lag features
   - Parse HOME/AWAY/STRK strings

3. **Prepare for Models (Day 2)**:
   - Generate hyperparameter grids
   - Export to DeepNote
   - Start Model 1 & 2

4. **Advanced Modeling (Day 3-7)**:
   - Implement ELO from scratch
   - Tune XGBoost
   - Build ensemble
   - Validate on holdout set

5. **Polish (Final days)**:
   - Uncertainty quantification
   - Visualizations
   - Presentation deck
