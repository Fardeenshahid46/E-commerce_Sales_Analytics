import streamlit as st
import pandas as pd
import plotly.express as px
from src import data_processing as dp
from src import rfm_clustering as rfm_mod
from src import forecast as fc  

st.set_page_config(page_title="📊 E-commerce Sales Analytics",page_icon="🛒",layout="wide")

@st.cache_data
def load_data(path="sample_data.csv"):
    df = dp.load_data(path)
    df = dp.clean_and_feature_engineer(df)
    return df

st.sidebar.title("⚙️ Controls")
uploaded=st.sidebar.file_uploader("📂 Upload Kaggle `sample_data.csv`", type=["csv"])
if uploaded is None:
    st.sidebar.info("Using default `sample_data.csv` in project root")
    df=load_data()
else:
    df=load_data(path=uploaded) 

periods = st.sidebar.slider("⏳ Forecast days", 7, 180, 30) 
model_type = st.sidebar.selectbox("🤖 Model", ["RandomForest (fast)", "Prophet (if installed)"])
st.sidebar.markdown("---")
st.sidebar.write("📌 Built with Python, Pandas, scikit-learn, Streamlit")

st.title("🛍️ E-Commerce Sales Prediction & Customer Segmentation")
tab1, tab2, tab3 = st.tabs(["📄 Data Overview", "👥 Customer Segmentation", "📈 Forecasting"])
with tab1:
    st.subheader("🔍 Raw Data Sample")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📊 Sales Over Time")
    ts = dp.aggregate_sales(df)
    fig = px.line(ts, x="ds", y="y", title="Total Sales Over Time", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", df["customer_id"].nunique())
    col2.metric("Total Orders", df["order_id"].nunique())
    col3.metric("Total Revenue (£)", f"{df['TotalAmount'].sum():,.0f}")      
    
with tab2:
        st.subheader("👥 RFM Segmentation")
        rfm = rfm_mod.compute_rfm(df)
        rfm, kmeans = rfm_mod.rfm_kmeans(rfm, n_clusters=4)
        st.dataframe(rfm.head(), use_container_width=True)

        fig2 = px.pie(rfm, names="cluster", title="Customer Segments (count)", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.scatter_3d(
            rfm, x="recency", y="frequency", z="monetary",
            color="cluster", title="3D RFM Segmentation"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
with tab3:
        st.subheader("📈 Sales Forecasting")
        ts = dp.aggregate_sales(df)

        if model_type.startswith("Random"):
            model, maes = fc.train_random_forest(ts, lags=14)
            preds = fc.predict_rf_future(model, ts, periods=periods, lags=14)
            future_dates = pd.date_range(ts["ds"].iloc[-1] + pd.Timedelta(days=1), periods=periods)
            forecast_df = pd.DataFrame({"ds": future_dates, "yhat": preds})
            figf = px.line(forecast_df, x="ds", y="yhat", title="RandomForest Forecast", template="plotly_white")
            st.plotly_chart(figf, use_container_width=True)
            st.info(f"Cross-Validation MAEs: {maes}")    
        else:
            if not fc.PROPHET_INSTALLED:
                st.error("⚠️ Prophet not installed. Use RandomForest or install prophet.")    
            else:
                m, forecast = fc.train_prophet(ts, periods=periods)
                figp = px.line(forecast, x="ds", y="yhat", title="Prophet Forecast", template="plotly_white")

                st.plotly_chart(figp, use_container_width=True)    
