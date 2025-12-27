FoodSight PH: Filipino Food Price Forecasting
Predictive Modeling using Time Series Regression (SARIMAX) > A Decision Support System for Price Volatility Mitigation

📖 Overview
FoodSight PH is a machine learning-based dashboard designed to forecast the monthly retail prices of 20 essential agricultural commodities in the Philippines.

Addressing the issue of food price volatility, this system utilizes a Hybrid Time Series Strategy, dynamically selecting between SARIMAX (Seasonal AutoRegressive Integrated Moving Average with Exogenous Variables) and Pure SARIMA models. It integrates historical price data from the World Food Programme (WFP) with key economic indicators to provide accurate forecasts up to 2027.

Key Objectives
Forecast: Predict future prices for staple foods (Rice, Meat, Vegetables, etc.).

Analyze: Visualize seasonal trends and price cycles.

Support: Provide data-driven insights for policy-making and consumer awareness.

🏗️ Architecture & Methodology
The system relies on historical data spanning from 2000 to 2025. It uses the following exogenous variables to improve prediction accuracy:

GWPI (General Wholesale Price Index)

Inflation Rate (Year-on-Year)

Global Fuel Prices (Brent Crude & Diesel)

Commodities Tracked
The model tracks 20 specific items, including:

Cereals: Rice (Regular/Milled), Maize (White/Yellow)

Meats: Chicken, Pork, Beef

Vegetables: Onions (Red), Garlic, Cabbage, Tomatoes, Potatoes

Fruits: Bananas (Lakatan/Saba), Mangoes

Others: Eggs, Sugar, Coconut, Fish (Bangus/Tilapia)

📂 Project Structure
Plaintext

FoodSight PH/
│
├── models/                       # Pre-trained SARIMAX models (.joblib)
│   ├── Bananas_(lakatan)_SARIMAX_model.joblib
│   ├── Rice_(regular,_milled)_SARIMAX_model.joblib
│   └── ... (20 total models)
│
├── app.py                        # Main Streamlit Dashboard application
├── dashboard_data.json           # Cached forecast data for fast loading
├── sarimax_final_training_data_complete_updated.csv  # Training dataset with economic indicators
├── wfp_food_prices_phl (main).csv # Raw WFP price dataset
├── requirements.txt              # Python dependencies
├── banner.png                    # Project Banner image
└── README.md                     # Project Documentation
🚀 Installation & Usage
Prerequisites
Python 3.8+

Git

Steps
Clone the repository

Bash

git clone https://github.com/yourusername/foodsight-ph.git
cd foodsight-ph
Install dependencies It is recommended to use a virtual environment.

Bash

pip install -r requirements.txt
Run the Dashboard

Bash

streamlit run app.py
Access the App The application will open automatically in your browser at http://localhost:8501.

🛠️ Technologies Used
Frontend: Streamlit (Web Framework)

Data Processing: Pandas, NumPy

Visualization: Plotly (Interactive Charts)

Machine Learning:

statsmodels (SARIMAX/ARIMA implementation)

pmdarima (Auto-ARIMA for parameter tuning)

scikit-learn (Metrics: MAE, RMSE, MAPE)

joblib (Model serialization)

📊 Data Sources
This project utilizes open data from:

World Food Programme (WFP): Global Food Prices Database (Philippines).

Philippine Statistics Authority (PSA): CPI and Inflation rates.

IndexMundi / World Bank: Global fuel prices.

👥 Authors
Department of Computer Engineering University of Science and Technology of Southern Philippines (USTP)

Reggie M. Abrera

Vhon Lorence C. Cabiluna

Joebert E. Zarate

📜 License
This project is intended for academic and educational purposes. Copyright © 2025 FoodSight PH Team.
