# Hockey Data Preprocessing Plan
# Run this when data is released

## Step 1: Initial Data Quality Check
ruby cli.rb diagnose data/raw/nhl_standings.csv > reports/initial_quality.txt
ruby cli.rb validate data/raw/nhl_standings.csv
ruby cli.rb profile data/raw/nhl_standings.csv

## Step 2: Data Cleaning
# Clean with side-by-side comparison to review changes
ruby cli.rb clean data/raw/nhl_standings.csv -o data/processed/nhl_clean.csv

## Step 3: Feature Engineering

# Rate statistics (per-game metrics)
ruby cli.rb rate data/processed/nhl_clean.csv GF GP -n goals_per_game
ruby cli.rb rate data/processed/nhl_clean.csv GA GP -n goals_against_per_game
ruby cli.rb rate data/processed/nhl_clean.csv PTS GP -n points_per_game

# Rolling averages (form/momentum over last N games)
ruby cli.rb rolling data/processed/nhl_clean.csv GF -w 5 -g team_name -o data/features/nhl_rolling5.csv
ruby cli.rb rolling data/features/nhl_rolling5.csv GF -w 10 -g team_name
ruby cli.rb rolling data/features/nhl_rolling5.csv GA -w 10 -g team_name
ruby cli.rb rolling data/features/nhl_rolling5.csv DIFF -w 10 -g team_name

# EWMA (recent form weighted more heavily)
ruby cli.rb ewma data/features/nhl_rolling5.csv DIFF -s 10 -g team_name
ruby cli.rb ewma data/features/nhl_rolling5.csv PTS -s 5 -g team_name

# Lag features (previous game stats for time-series models)
ruby cli.rb lag data/features/nhl_rolling5.csv W 1,3,5 -g team_name
ruby cli.rb lag data/features/nhl_rolling5.csv DIFF 1,3,5 -g team_name
ruby cli.rb lag data/features/nhl_rolling5.csv PTS 1 -g team_name

# Cumulative stats (season progress)
ruby cli.rb cumulative data/features/nhl_rolling5.csv W -s sum -g team_name -o data/features/nhl_full.csv
ruby cli.rb cumulative data/features/nhl_full.csv PTS -s sum -g team_name

# Rankings (where team stands)
ruby cli.rb rank data/features/nhl_full.csv PTS -g division
ruby cli.rb rank data/features/nhl_full.csv DIFF

# One-hot encode categorical features
ruby cli.rb encode data/features/nhl_full.csv division -t onehot
ruby cli.rb encode data/features/nhl_full.csv conference -t onehot

# Normalize key features for linear regression
ruby cli.rb normalize data/features/nhl_full.csv GF -m minmax -o data/features/nhl_final.csv
ruby cli.rb normalize data/features/nhl_final.csv GA -m minmax
ruby cli.rb normalize data/features/nhl_final.csv PTS -m zscore

## Step 4: Generate Hyperparameter Grids
ruby cli.rb hyperparam-grid config/hyperparams/model2_linear_regression.yaml -o experiments/lr_grid.csv
ruby cli.rb hyperparam-random config/hyperparams/model4_xgboost.yaml 50 -o experiments/xgb_random.csv
ruby cli.rb hyperparam-random config/hyperparams/model4_random_forest.yaml 30 -o experiments/rf_random.csv
ruby cli.rb hyperparam-grid config/hyperparams/model5_ensemble.yaml -o experiments/ensemble_grid.csv

## Step 5: Export for DeepNote
# Final dataset ready for modeling
cp data/features/nhl_final.csv data/for_deepnote/hockey_features.csv

# Git commit
git add data/features/ data/for_deepnote/ experiments/
git commit -m "Preprocessed NHL standings data with engineered features"
git push

echo "Data preprocessing complete! Ready for DeepNote modeling."
