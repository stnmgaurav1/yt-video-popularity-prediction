# Student Guide: YouTube Video Popularity Prediction

## Problem statement
Creators want to understand how video properties relate to engagement. We predict video likes from views, duration, and number of tags.

## Key definitions
- **Regression**: predict a number, such as 1,250 likes.
- **Features (X)**: views, duration in seconds, tag count.
- **Target (y)**: likes.
- **Random Forest Regressor**: an ensemble of decision trees whose predictions are averaged.
- **RMSE**: square root of the mean of squared differences between actual and predicted likes; lower is better. It is in units of likes.
- **MAE**: average absolute prediction error in likes.
- **R²**: how much variation in target is explained compared with predicting its average; it may be negative.

## Workflow
CSV → Clean data → Select X and y → Train/test split → Random Forest → Predict → RMSE → Streamlit dashboard.

## Demonstration
1. Install Python 3.10+ and VS Code.
2. Open this extracted project folder in VS Code.
3. Open Terminal → New Terminal.
4. Run the setup and launch commands from README.md.
5. In the browser enter views=10000, duration=300, tags=5 and click Predict popularity.
6. Show outputs/metrics.csv and outputs/feature_importance.png to your professor.

## Viva questions
1. **Why regression?** Likes is a continuous numerical target.
2. **Why Random Forest?** It models nonlinear patterns and interactions with little feature scaling.
3. **What is RMSE?** Square root of mean squared prediction errors, in likes.
4. **Why split the data?** To evaluate on records not used in training.
5. **What does feature importance show?** Which inputs the model uses most; it does not prove causation.
6. **Why log-transform likes?** Engagement counts are often heavily skewed; log1p helps training.
7. **What is overfitting?** The model learns training-specific patterns that do not generalize.
8. **Can it predict popularity before uploading?** No, views are not available before uploading.
9. **What is a limitation?** Views and likes may both be driven by channel size, promotion, and age of the video.
10. **How can we improve?** Use real data, time-based validation, channel and publication metadata, and pre-upload features for genuine forecasting.

## Suggested 7-slide presentation
1. Title and objective
2. Problem statement and input/target
3. Dataset and preprocessing
4. Random Forest workflow
5. Evaluation with RMSE and charts
6. Streamlit demo
7. Limitations, content strategy and conclusion

## Academic integrity
The included dataset is synthetic. Do not claim it is a Kaggle dataset. Record the Kaggle dataset title, URL, license, sample count and measured RMSE after training on a real dataset.
