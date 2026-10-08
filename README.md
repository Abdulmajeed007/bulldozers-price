# Bulldozer Price Predictor 🚜

A machine learning app that predicts the sale price of heavy equipment (bulldozers) from a CSV of sale records.

**Live demo:** https://bulldozers-price.streamlit.app

## What it does
Upload a CSV of bulldozer sales in the Kaggle "Blue Book for Bulldozers" format. The app cleans the data, runs it through the trained model, and shows a `PredictedPrice` for every row. If your file also includes `SalePrice`, it shows the average error against the actual prices.

A sample file is available from the app ("Download a sample CSV to try").

## Model
- Hyperparameter-tuned Random Forest Regressor (scikit-learn 1.6.1)
- Trained on the Kaggle Blue Book for Bulldozers dataset
- Validation score: [add your RMSLE or R² here]

## Preprocessing (applied in the app exactly as in training)
1. Split `saledate` into year, month, day, day of week and day of year
2. Fill missing numeric values with the training median, with an `_is_missing` flag column
3. Convert text columns to numeric category codes, with an `_is_missing` flag column
4. Arrange the 102 features in the exact order the model expects

## Files
- `app.py`: Streamlit app
- `bulldozer_model (1).joblib`: trained model (compressed)
- `bulldozer_prep.json`: category codes, medians and feature order from training
- `sample_bulldozers.csv`: example input
- `requirements.txt`: dependencies

## Limitations
- The sample file comes from the training data, so its error looks better than real-world performance. Use the validation score for an honest estimate.
- Input must follow the Kaggle column format, and the app handles up to 1,000 rows at a time.
- Built as a learning and portfolio project.

## Author
Sayudi Mohammed Abdulmajeed
