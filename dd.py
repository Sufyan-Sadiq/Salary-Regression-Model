import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- 1. Data Generation and Preprocessing ---
# This creates a synthetic dataset for demonstration purposes.
# In a real-world scenario, you would load your data from a CSV or database.
st.write("---")
st.header("1. Model Training Process")
st.write("Generating synthetic data for the salary regression model...")

np.random.seed(42)
data = {
    'Years_of_Experience': np.random.uniform(1, 20, 1000).astype(int),
    'Education_Level': np.random.choice(['Bachelor', 'Master', 'PhD'], 1000),
    'City': np.random.choice(['New York', 'London', 'Berlin', 'Tokyo'], 1000),
}
df = pd.DataFrame(data)

# Create a salary column with some noise, dependent on other features
df['Salary'] = (
    15000 +
    df['Years_of_Experience'] * 5000 +
    df['Education_Level'].apply(lambda x: {'Bachelor': 10000, 'Master': 25000, 'PhD': 50000}[x]) +
    df['City'].apply(lambda x: {'New York': 8000, 'London': 5000, 'Berlin': 3000, 'Tokyo': 7000}[x]) +
    np.random.normal(0, 10000, 1000)
)

# Preprocessing the data
# Use LabelEncoder for categorical features
le_education = LabelEncoder()
le_city = LabelEncoder()

df['Education_Level_Encoded'] = le_education.fit_transform(df['Education_Level'])
df['City_Encoded'] = le_city.fit_transform(df['City'])

# Define features and target
features = df[['Years_of_Experience', 'Education_Level_Encoded', 'City_Encoded']]
target = df['Salary']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. Build and Train the ANN Model ---
st.write("Building and training the Artificial Neural Network...")

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
st.write("Model training complete!")
st.write("---")

# --- 3. Streamlit App Interface ---
st.title("Salary Regression Predictor")
st.markdown(
    """
    This application predicts an individual's salary based on years of experience,
    education level, and city, using an Artificial Neural Network (ANN).
    """
)

# User inputs
st.subheader("Enter Your Information")
years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
education = st.selectbox("Education Level", le_education.classes_)
city = st.selectbox("City", le_city.classes_)

# Prediction button
if st.button("Predict Salary"):
    try:
        # Preprocess user input
        education_encoded = le_education.transform([education])[0]
        city_encoded = le_city.transform([city])[0]

        # Create a DataFrame for the user's input
        input_data = pd.DataFrame([[years_exp, education_encoded, city_encoded]],
                                  columns=['Years_of_Experience', 'Education_Level_Encoded', 'City_Encoded'])

        # Scale the input data using the same scaler used for training
        input_scaled = scaler.transform(input_data)

        # Make the prediction
        predicted_salary = model.predict(input_scaled)[0][0]

        # Display the result
        st.success(f"### The predicted salary is: ${predicted_salary:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- 4. How to run this app ---
st.write("---")
st.subheader("How to Run This App")
st.info(
    """
    1.  **Save the code:** Save the code above into a file named `salary_app.py`.
    2.  **Install libraries:** Open your terminal or command prompt and run the following command to install all required libraries:
        ```
        pip install streamlit numpy pandas scikit-learn tensorflow
        ```
    3.  **Run the app:** From the same directory where you saved `salary_app.py`, run this command:
        ```
        streamlit run salary_app.py
        ```
    4.  **View in browser:** Your web browser will open a new tab with the application.
    """
)
