import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Salary Prediction App", page_icon="💼")
st.title("Salary Prediction App")
st.markdown("This app predicts your salary based on company experience.")

# cute animated gif
st.image("https://media.giphy.com/media/3o7aD2saq1z5b6d0g4/giphy.gif", caption="Let's predict your salary!")

# divider
st.divider()

# inputs
col1, col2, col3 = st.columns(3)

with col1:
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=20, value=3)

with col2:
    satisfaction_level = st.slider("Satisfaction Level", min_value=0.0, max_value=1.0,step=0.01, value=0.5)

with col3:
    average_monthly_hours = st.slider("Average Monthly Hours", min_value=120, max_value=310, value=160)

x = [years_at_company, satisfaction_level, average_monthly_hours]

# Load model and Scalar
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Predict button

predict_button = st.button("Predict Salary")


st.divider()

if predict_button:
    st.balloons()

    x_array = scaler.transform([np.array(x)])
    prediction = model.predict(x_array)

    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")

    # Visualize user input
    df_input = pd.DataFrame({
        "Feature": ["Years at Company", "Satisfaction Level", "Average Monthly Hours"],
        "Value": x
    })

    fig = px.bar(df_input, x="Feature", y="Value", color="Feature",
                 title="User Input Profile", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please enter your details and click on 'Predict Salary' to see the prediction.")

