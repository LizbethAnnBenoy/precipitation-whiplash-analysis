# precipitation-whiplash-analysis
This project uses data mining and machine learrning techniques for pattern analysis, and prediction of precipitation whiplash events, which is the rapid transition from severe drought to extreme flooding.
It combines data warehousing, pattern mining and ML to get patterns and predict events like this if possible.

## Objective
- Collect data
- Preprocess it and make a data warehouse
- Identify patterns leading to whiplash events
- Apply association rule mining and clustering
- Build machine learning models for prediction
- Visualize trends and seasonal behavior
- Develop an interactive user interface

## Dataset
The dataset includes:
- Precipitation (mm/day)
- IVT (Atmospheric River indicator)
- Drought severity score
- Soil moisture (dry land memory)
- Binary indicators:
- Atmospheric River (is_ar)
- Extreme precipitation (extreme_precip)
- Whiplash event (is_whiplash)

## Methods Used
- Data Preprocessing: Handling missing values, feature engineering
- Pattern Mining: Apriori algorithm for association rules
- Clustering: K-Means for event categorization
- Machine Learning:
- Random Forest
- Gradient Boosting
- Visualization: Time series, heatmaps, distributions
- Dashboard: Gradio-based interactive UI
