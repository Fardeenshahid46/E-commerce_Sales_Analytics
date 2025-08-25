# E-commerce_Sales_Analytics
This project is a full end-to-end data science workflow for analyzing e-commerce transactions, segmenting customers, and predicting future sales using Machine Learning & Time Series models. It combines Python scripts, Jupyter notebooks, and a Streamlit app into a single pipeline, reflecting real industry practices.

🚀 Features
🔍 Data Processing

  Cleans raw Kaggle UK Online Retail dataset (canceled orders, missing IDs, etc.).
  
  Feature engineering: total sales amount, temporal features, customer-level summaries.

📊 Exploratory Data Analysis (EDA)

  Sales trends over time.
  
  Top customers & products.
  
  Key metrics (customers, orders, revenue).

👥 Customer Segmentation

  RFM Analysis (Recency, Frequency, Monetary).
  
  K-Means clustering to group customers into segments.
  
  Interactive 3D scatter visualization and cluster proportions.

📈 Sales Forecasting

  RandomForest time-series regression with lag features.
  
  Optional Prophet model for interpretable forecasts.
  
  Cross-validation with MAE scoring.
  
  Interactive future prediction charts (7–180 days).

🖥️ Streamlit Dashboard

  Modern tab-based interface with KPIs, charts, and cluster visuals.
🎯 Use Cases
  ✔️ Retail analytics teams for sales forecasting
  ✔️ Marketing for customer segmentation & targeting
  ✔️ Business analysts for KPI monitoring
  ✔️ Data science students to learn end-to-end ML pipelines
    
    Sidebar controls: dataset upload, forecast horizon, model choice.

Interactive Plotly charts for real-time insights.
