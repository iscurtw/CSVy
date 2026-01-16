# Competition Dataset - Confirmed Features

## Raw Features Available

### Numerical (19 features):
1. **GP** - Games Started
2. **RW** - Regulation Wins (2 points)
3. **OTW** - Overtime Wins (2 points)
4. **RL** - Regulation Losses (0 points)
5. **OTL** - Overtime Losses (1 point)
6. **PTS** - Total Points
7. **P%** - Points Percentage
8. **GF** - Goals For
9. **GA** - Goals Against
10. **DIFF** - Goal Differential (GF - GA)
11. **HOME** - Home Record
12. **AWAY** - Away Record
13. **OT** - OT Record
14. **L10** - Recent Record (Last 10 Games)
15. **STRK** - Win Streak
16. **travel_time** - Travel Time (hours)
17. **travel_distance** - Travel Distance (miles)
18. **rest_time** - Days Since Last Game
19. **injuries** - Number of Key Injuries

### Categorical (5 features):
1. **clinch_status** - Values: `playoff`, `division`, `conference`, `none`
2. **award** - Presidents' Trophy (best regular season record)
3. **elimination** - Playoff elimination status: `eliminator`, `eliminated`, `active`
4. **travel** - Away team traveling to home court (binary)
5. **division** - Team tier: `D1`, `D2`, `D3`

---

## Feature Engineering Priorities

### Tier 1: Use Built-Ins Directly
```python
features = ['rest_time', 'travel_distance', 'travel_time', 'injuries', 
            'GF', 'GA', 'DIFF', 'P%', 'STRK', 'L10']
```

### Tier 2: Create Interaction Features
```python
# Fatigue index (travel + no rest = disaster)
df['fatigue_index'] = df['travel_distance'] / (df['rest_time'] + 1)

# Injury severity when traveling
df['injuries_away'] = df['injuries'] * df['travel']

# Clinch motivation (clinched teams may coast)
df['clinched_late_season'] = (df['clinch_status'] != 'none') & (df['GP'] > 60)

# Elimination tanking
df['eliminated_tanking'] = (df['elimination'] == 'eliminated') & (df['GP'] > 50)

# Division strength (D1 teams should dominate)
df['division_rank'] = df['division'].map({'D1': 3, 'D2': 2, 'D3': 1})
```

### Tier 3: Rolling/Lag Features
```python
# Recent form
df['GF_rolling_5'] = df.groupby('team')['GF'].rolling(5).mean()
df['rest_time_avg_5'] = df.groupby('team')['rest_time'].rolling(5).mean()

# Momentum shifts
df['STRK_change'] = df.groupby('team')['STRK'].diff()
df['injuries_trend'] = df.groupby('team')['injuries'].diff()
```

---

## How Each Model Uses Features

### Model 1: Baseline
- Just `DIFF` and `P%` (simple average predictor)

### Model 2: Linear Regression
- All numerical features normalized
- One-hot encode: `division`, `clinch_status`, `elimination`
- Polynomial features: `DIFF^2`, `GF*GA`, `rest_time*travel_distance`

### Model 3: ELO
- W/L/OT outcomes
- `rest_time` → feeds into `rest_advantage_per_day`
- `travel_distance` → feeds into `b2b_penalty` equivalent
- `HOME` flag → feeds into `home_advantage`
- `injuries` → dynamic K-factor adjustment

### Model 4: XGBoost/Random Forest
- **Everything** (all 24 raw + 20 engineered features)
- Trees automatically find: "If travel_distance > 500 AND rest_time < 2 AND injuries > 1 → bad"
- Feature importance will reveal: travel_distance, rest_time, DIFF, injuries as top predictors

### Model 5: Ensemble
- Predictions from Models 1-4
- Weight based on recent performance
- `injuries` and `travel` as meta-features (when do models disagree?)

---

## Key Insights from Feature Set

🔥 **Travel/Rest/Injuries are built-in!** - This is huge. Most competitions require manual collection.

🏆 **Clinch/Elimination psychology** - Captures motivation effects (teams that clinched early may coast, eliminated teams tank)

⚡ **Presidents' Trophy curse** - Historical stat: award winners often underperform expectations

📊 **Division tiers (D1/D2/D3)** - Suggests this is multi-level hockey (not just NHL), so skill gaps are large

💡 **L10 + STRK** - Recent form indicators already calculated (momentum)

---

## Recommended Feature Priority

**Must-have (40% of predictive power):**
- DIFF, GF, GA, P%
- rest_time, travel_distance, injuries
- division (D1 vs D3 is a huge skill gap)

**High-value (30%):**
- STRK, L10 (momentum)
- clinch_status, elimination (motivation)
- HOME, AWAY splits
- fatigue_index (interaction)

**Nice-to-have (20%):**
- travel_time (redundant with distance)
- award (Presidents' Trophy curse)
- OT record (tiebreaker situations)

**Experimental (10%):**
- Weather data (external scraping)
- Coaching changes (manual research)
- Rivalry games (historical head-to-head)

---

## Next Steps

1. ✅ **Run diagnostics** on actual dataset when released
2. ✅ **Normalize** travel_distance, travel_time, injuries (scale 0-1)
3. ✅ **One-hot encode** division, clinch_status, elimination
4. ✅ **Create interactions** (fatigue_index, injuries_away, etc.)
5. ✅ **Generate hyperparameter grids** for all 5 models
6. 🚀 **Train in DeepNote**, track results in HTML reports
