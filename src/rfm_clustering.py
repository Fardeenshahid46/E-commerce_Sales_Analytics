import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def compute_rfm(df,ref_date=None):
    df=df.copy()
    if ref_date is None:
       ref_date=df["order_date"].max() + pd.Timedelta(days=1)  
    
    rfm=df.groupby('customer_id').agg({
        "order_date":lambda x:(ref_date - x.max()).days,
        'order_id':'nunique',
        "TotalAmount":"sum"
    }).reset_index()
    rfm.columns=["customer_id","recency","frequency","monetary"]
    rfm["monetary"]=rfm["monetary"].clip(lower=0.01)
    return rfm     

def rfm_kmeans(rfm,n_clusters=4,random_state=42):
    features=["recency","frequency","monetary"]
    X=rfm[features].copy()
    
    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)
    
    kmeans=KMeans(n_clusters=n_clusters,random_state=random_state)
    rfm['cluster']=kmeans.fit_predict(Xs)
    
    cluster_info=pd.DataFrame(kmeans.cluster_centers_,columns=features)
    cluster_info=scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_info=pd.DataFrame(cluster_info,columns=features)
    return rfm, cluster_info

if __name__ == "__main__":
    import data_processing as dp
    df=dp.load_data()
    df=dp.clean_and_feature_engineer(df)
    rfm=compute_rfm(df)
    rfm, kmeans=rfm_kmeans(rfm)
    print(rfm.head())