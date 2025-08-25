import matplotlib.pyplot as plt

def plot_sales_ts(ts,ax=None):
    if ax is None:
        ax=plt.gca()
    ax.plot(ts['ds'],ts['y'])
    ax.set_title("Total Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    return ax

def top_n_customers(df,n=10):
    s=df.groupby('customer_id')["TotalAmount"].sum().sort_values(ascending=False).head(n)
    return s
