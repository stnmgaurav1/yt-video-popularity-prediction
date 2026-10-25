# YouTube Video Popularity Prediction (Regression)

## Objective
Predict the number of likes (a measurable proxy for popularity) from video views, duration in seconds, and tag count using a Random Forest Regressor. Evaluate using RMSE; MAE and R² are additional metrics.

## Folder structure
- `data/youtube_sample.csv`: **synthetic practice data**, not downloaded from Kaggle.
- `src/pipeline.py`: data cleaning, training, evaluation and prediction.
- `train.py`: training command.
- `app.py`: interactive Streamlit interface.
- `models/`: saved trained model.
- `outputs/`: metrics, predictions and graphs.
- `PROJECT_GUIDE.md`: setup, concepts and viva questions.

## Run on Windows (VS Code terminal)
```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```
If activation is blocked, use `.venv\Scripts\python.exe` instead of `python` for each command and `.venv\Scripts\python.exe -m streamlit run app.py`.

## Use Kaggle data
Download a YouTube Statistics CSV from Kaggle and save it in `data/`. It must contain **views**, **likes**, **duration**, and **tags** (common variants like `view_count` and `like_count` work). Run:
```powershell
python train.py --data "data/your_kaggle_file.csv"
```
Not all Kaggle YouTube Statistics datasets contain duration or tags. If absent, use a compatible dataset or join valid duration information by video ID; do not invent missing fields.

## Methodology
1. Load CSV and standardize supported column names.
2. Parse duration, count tags, remove invalid rows and duplicates.
3. Split 80% training and 20% test with fixed random state.
4. Fit Random Forest Regressor on log1p(likes) to stabilize highly skewed targets.
5. Transform predictions back to like counts and calculate RMSE, MAE and R² on original scale.
6. Display estimates and interpret feature importance.

## Caveats
Views are post-publication information, so this is an **engagement estimation** model, not a reliable pre-upload virality predictor. A sample dataset is provided solely to test the workflow. Replace it with real, licensed Kaggle data before claiming real-world accuracy. A random row split may overestimate generalization if repeated videos or channels appear in both sets. Tags count measures quantity, not meaning. Like counts are not the only definition of popularity.
