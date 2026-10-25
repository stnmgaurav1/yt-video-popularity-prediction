from pathlib import Path
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['views', 'duration_seconds', 'tag_count']

def duration_seconds(value):
    if pd.isna(value): return np.nan
    s = str(value).strip()
    if re.fullmatch(r'\d+(\.\d+)?', s): return float(s)
    if ':' in s:
        try:
            nums = [float(x) for x in s.split(':')]
            if len(nums) <= 3: return sum(v * 60**i for i, v in enumerate(reversed(nums)))
        except ValueError: pass
    m = re.fullmatch(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', s, re.I)
    if m: return sum(int(v or 0)*factor for v,factor in zip(m.groups(),[3600,60,1]))
    return np.nan

def tag_count(value):
    if pd.isna(value) or str(value).strip().lower() in ('','[none]','none','nan'): return 0
    s = str(value).strip()
    if '|' in s: return len([x for x in s.split('|') if x.strip()])
    if ',' in s: return len([x for x in s.split(',') if x.strip()])
    return len([x for x in s.split(' ') if x.strip()]) if ' ' not in s else 1

def load_dataset(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    lookup = {c.lower().strip().replace(' ','_'): c for c in df.columns}
    def column(*names):
        for name in names:
            if name in lookup: return df[lookup[name]]
        return None
    views = column('views','view_count','viewcount')
    likes = column('likes','like_count','likecount')
    duration = column('duration','duration_seconds','video_duration','length')
    tags = column('tags','video_tags','tag_count')
    missing = [name for name,val in [('views',views),('likes',likes),('duration',duration),('tags',tags)] if val is None]
    if missing: raise ValueError('Missing required columns: '+', '.join(missing)+'. Choose a Kaggle YouTube dataset with these columns.')
    tag_series = tags.apply(tag_count) if tags.name.lower().strip().replace(' ','_') != 'tag_count' else pd.to_numeric(tags,errors='coerce')
    clean = pd.DataFrame({'views':pd.to_numeric(views,errors='coerce'), 'duration_seconds':duration.apply(duration_seconds), 'tag_count':tag_series, 'likes':pd.to_numeric(likes,errors='coerce')})
    clean = clean.replace([np.inf,-np.inf],np.nan).dropna()
    clean = clean[(clean >= 0).all(axis=1)]
    clean = clean.drop_duplicates()
    if len(clean) < 20: raise ValueError('Need at least 20 valid records after cleaning.')
    return clean

def train(csv_path, save=True):
    df = load_dataset(csv_path)
    X, y = df[FEATURES], np.log1p(df['likes'])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor(n_estimators=150,min_samples_leaf=2,random_state=42,n_jobs=-1)
    model.fit(X_train,y_train)
    predictions = np.maximum(0,np.expm1(model.predict(X_test)))
    actual = np.expm1(y_test)
    metrics = {'RMSE':float(np.sqrt(mean_squared_error(actual,predictions))), 'MAE':float(mean_absolute_error(actual,predictions)), 'R2':float(r2_score(actual,predictions)), 'train_rows':len(X_train), 'test_rows':len(X_test)}
    if save:
        (ROOT/'models').mkdir(exist_ok=True)
        (ROOT/'outputs').mkdir(exist_ok=True)
        joblib.dump(model,ROOT/'models'/'random_forest.joblib')
        pd.DataFrame([metrics]).to_csv(ROOT/'outputs'/'metrics.csv',index=False)
        pd.DataFrame({'actual_likes':actual.values,'predicted_likes':predictions}).to_csv(ROOT/'outputs'/'predictions.csv',index=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,5))
        plt.scatter(actual,predictions,alpha=.5)
        upper=max(float(max(actual)),float(max(predictions)),1)
        plt.plot([0,upper],[0,upper],'--',color='gray')
        plt.xlabel('Actual likes');plt.ylabel('Predicted likes');plt.title('Actual vs Predicted Likes');plt.tight_layout()
        plt.savefig(ROOT/'outputs'/'actual_vs_predicted.png',dpi=140);plt.close()
        plt.figure(figsize=(7,4))
        plt.bar(FEATURES,model.feature_importances_)
        plt.title('Feature Importance');plt.ylabel('Importance');plt.tight_layout()
        plt.savefig(ROOT/'outputs'/'feature_importance.png',dpi=140);plt.close()
    return model, metrics, df

def predict(model,views,duration,tag_count_value):
    x = pd.DataFrame([[views,duration,tag_count_value]],columns=FEATURES)
    return max(0,int(round(np.expm1(model.predict(x)[0]))))
