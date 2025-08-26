import pandas as pd

CONFIG = {
    'kaggle_path': 'data_sample.csv'
}

def load_data(path=None):
    path = path or CONFIG['kaggle_path']
    df = pd.read_csv(path, encoding='latin1', parse_dates=["InvoiceDate"])
    return df

def clean_and_feature_engineer(df):
    df = df.copy()
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df.dropna(subset=["CustomerID"])
    df["Quantity"] = df["Quantity"].astype(int)
    df["UnitPrice"] = df["UnitPrice"].astype(float)
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    
    df = df.rename(columns={
      "InvoiceNo": "order_id",
      "CustomerID": "customer_id",
      "InvoiceDate": "order_date",
      "UnitPrice": "price",
      "Quantity": "quantity"
    })
    
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_day"] = df["order_date"].dt.day
    df["order_weekday"] = df["order_date"].dt.weekday
    return df

def aggregate_sales(df, freq="D"):
    ts = df.groupby("order_date")["TotalAmount"].sum().rename('y').to_frame()
    ts = ts.resample(freq).sum().fillna(0).reset_index().rename(columns={"order_date": "ds"})
    return ts

if __name__ == "__main__":
    df = load_data()
    df = clean_and_feature_engineer(df)
    print("Rows after cleaning:", len(df))
    print(df.head())
