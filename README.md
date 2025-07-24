# Restaurant Sales Forecasting

This project forecasts restaurant sales based on historical data. It supports forecasting for:
 **Next week**
 **Next month**

## Dataset
- The dataset was sourced from [Kaggle](https://www.kaggle.com/) and preprocessed to ensure quality and relevance.
- Unnecessary columns were removed, and new features were engineered to enhance model performance.

## Data Processing
- Feature engineering was performed to extract meaningful insights.
- Irrelevant features were dropped to reduce noise.
- Feature importance was evaluated to retain significant predictors only.

## Machine Learning Models
Several models were tested, but the following three were selected for detailed comparison:
- **XGBoost**
- **LightGBM**
- **CatBoost**

### Evaluation Metrics Used:
- MAPE (Mean Absolute Percentage Error)
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)

### Visual Comparison
Model performance was also compared using graphs for better interpretability.

## Best Model
Based on the metrics and visual comparisons, **XGBoost** emerged as the best performing model.

##  Report
The complete report—including data cleaning steps, feature engineering process, and model comparisons—is available as a PDF:

[`forecasting_report.pdf`](./forecasting_report.pdf)

