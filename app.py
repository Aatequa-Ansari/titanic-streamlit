import streamlit as st
import pandas as pd
import pickle

# Load trained model pipeline
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.title("Titanic Survival Prediction System")

st.write(
    """
    This application predicts whether a passenger would have survived
    the Titanic disaster based on personal and travel details.
    """
)

# Sidebar for inputs
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox(
    "Passenger Class",
    [1, 2, 3],
    help="1 = Upper class, 2 = Middle class, 3 = Lower class"
)

sex = st.sidebar.selectbox(
    "Sex",
    ["male", "female"]
)

age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=30
)

sibsp = st.sidebar.number_input(
    "Number of Siblings / Spouses",
    min_value=0,
    max_value=8,
    value=0
)

parch = st.sidebar.number_input(
    "Number of Parents / Children",
    min_value=0,
    max_value=6,
    value=0
)

fare = st.sidebar.number_input(
    "Ticket Fare",
    min_value=0.0,
    max_value=600.0,
    value=30.0
)

embarked = st.sidebar.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"],
    help="S = Southampton, C = Cherbourg, Q = Queenstown"
)

# Create input DataFrame with exact training columns
input_df = pd.DataFrame({
    "pclass": [pclass],
    "sex": [sex],
    "age": [age],
    "sibsp": [sibsp],
    "parch": [parch],
    "fare": [fare],
    "embarked": [embarked]
})

# Prediction section
st.subheader("Prediction Result")

if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("Prediction: Passenger would SURVIVE")
    else:
        st.error("Prediction: Passenger would NOT survive")

    st.write(f"Survival Probability: {probability:.2f}")
