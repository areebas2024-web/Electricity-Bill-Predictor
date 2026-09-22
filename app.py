import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Electricity Bill Predictor",
    page_icon="⚡",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⚡ Electricity Bill Predictor")
st.write("Predict your monthly electricity bill using Machine Learning.")


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("electricity_bill_dataset.csv")


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("ElectricityBill", axis=1)
y = df["ElectricityBill"]


# --------------------------------------------------
# CATEGORICAL AND NUMERICAL COLUMNS
# --------------------------------------------------

categorical_columns = ["City", "Company"]

numerical_columns = [
    "Fan",
    "Refrigerator",
    "AirConditioner",
    "Television",
    "Monitor",
    "MotorPump",
    "Month",
    "MonthlyHours",
    "TariffRate"
]


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)


# --------------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("R² Score", f"{r2:.2f}")


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.subheader("🔢 Enter Your Details")


fan = st.number_input(
    "Fan",
    min_value=0,
    value=5
)

refrigerator = st.number_input(
    "Refrigerator",
    min_value=0,
    value=10
)

air_conditioner = st.number_input(
    "Air Conditioner",
    min_value=0,
    value=2
)

television = st.number_input(
    "Television",
    min_value=0,
    value=5
)

monitor = st.number_input(
    "Monitor",
    min_value=0,
    value=2
)

motor_pump = st.number_input(
    "Motor Pump",
    min_value=0,
    value=1
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=6
)

city = st.selectbox(
    "City",
    sorted(df["City"].unique())
)

company = st.selectbox(
    "Electricity Company",
    sorted(df["Company"].unique())
)

monthly_hours = st.number_input(
    "Monthly Hours",
    min_value=0,
    value=300
)

tariff_rate = st.number_input(
    "Tariff Rate",
    min_value=0.0,
    value=8.0
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("⚡ Predict Electricity Bill"):

    new_data = pd.DataFrame({
        "Fan": [fan],
        "Refrigerator": [refrigerator],
        "AirConditioner": [air_conditioner],
        "Television": [television],
        "Monitor": [monitor],
        "MotorPump": [motor_pump],
        "Month": [month],
        "City": [city],
        "Company": [company],
        "MonthlyHours": [monthly_hours],
        "TariffRate": [tariff_rate]
    })

    prediction = model.predict(new_data)[0]

    st.success(
        f"💰 Estimated Electricity Bill: ₹{prediction:,.2f}"
    )


# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

with st.expander("📁 View Dataset"):

    st.dataframe(df)

st.caption("Machine Learning Model: Linear Regression")
