import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import data_processing as dp
from src import rfm_clustering as rfm_mod
from src import forecast as fc

raw=dp.load_data()
df=dp.clean_and_feature_engineer(raw)
print("rows:",len(df))

print(df.describe(include='all'))

ts=dp.aggregate_sales(df)
plt.figure(figsize=(12,4))
plt.plot(ts['ds'],ts['y'])
plt.title("Total Sales Over Time")
plt.show()

print("Top Customers:")
print(dp.top_n_customers(df))
print(df.groupby("customer_id")["TotalAmount"].sum().sort_values(ascending=False).head(10))

rfm=rfm_mod.compute_rfm(df)
rfm,kmeans=rfm_mod.rfm_kmeans(rfm,n_clusters=4)
print(rfm.groupby('cluster')['customer_id'].count())

ts=fc.prepare_time_series(df)
model,maes=fc.train_random_forest(ts,lags=14)
print("CV MAEs:",maes)

preds=fc.predict_rf_future(model,ts,periods=30,lags=14)
print(preds[:10])