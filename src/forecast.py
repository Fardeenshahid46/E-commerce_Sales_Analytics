import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

try:
    from prophet import Prophet
    PROPHET_INSTALLED = True
except ImportError:
    PROPHET_INSTALLED = False
    
def prepare_time_series(df,date_col="order_date",value_col="TotalAmount",freq="D"):
    ts=df[[date_col,value_col]].copy()
    ts=ts.groupby(date_col).sum().rename(columns={value_col:"y"})
    ts=ts.resample(freq).sum().fillna(0).reset_index()
    ts.columns=["ds","y"]
    return ts

def train_prophet(ts,periods=30):
    if not PROPHET_INSTALLED:
        raise RuntimeError("Prophet is not installed.")
    model=Prophet()
    model.fit(ts)
    future=model.make_future_dataframe(periods=periods)
    forecast=model.predict(future)
    return model,forecast

def train_random_forest(ts,lags=14,n_estimators=100):
    df=ts.copy()
    for lag in range(1,lags+1):
        df[f'lag_{lag}']=df["y"].shift(lag)
    df=df.dropna().reset_index(drop=True)
    
    features=[c for c in df.columns if c.startswith('lag_')]
    X=df[features]
    y=df["y"]
    
    tscv=TimeSeriesSplit(n_splits=3)
    maes=[]
    model=RandomForestRegressor(n_estimators=n_estimators,random_state=42)
    
    for train_idx,test_idx in tscv.split(X):
        X_train,X_test=X.iloc[train_idx],X.iloc[test_idx]
        y_train,y_test=y.iloc[train_idx],y.iloc[test_idx]
        model.fit(X_train,y_train)
        preds=model.predict(X_test)
        maes.append(mean_absolute_error(y_test,preds))
        
    model.fit(X,y)
    return model,maes

def predict_rf_future(model,ts,periods=30,lags=14):
    last_vals=ts["y"].iloc[-lags:].tolist()
    preds=[]
    for _ in range(periods):
        feat=np.array(last_vals[-lags:]).reshape(1,-1) 
        p=float(model.predict(feat)[0])
        preds.append(p)
        last_vals.append(p)
    return preds

if __name__ == "__main__":
    import data_processing as dp    
    df=dp.load_data()
    df=dp.clean_and_feature_engineer(df)
    ts=prepare_time_series(df)
    model,maes=train_random_forest(ts)
    print("CV MAEs:",maes)
    print("Next 7 preds:",predict_rf_future(model,ts,periods=7))