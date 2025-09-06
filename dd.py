import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor

# -----------------------------
# Generate synthetic dataset
# -----------------------------
np.random.seed(42)
n = 500
years_exp = np.random.normal(6, 3, n).clip(0, 20)
education_levels = np.random.choice(["High School", "Bachelors", "Masters", "PhD"],
                                    size=n, p=[0.25, 0.5, 0.2, 0.05])
skill_score = np.random.uniform(0, 100, n)
age = (years_exp + np.random.normal(22, 4, n)).clip(18, 65)
location_idx = np.random.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])

edu_base = {"High School": 30000, "Bachelors": 45000,
            "Masters": 60000, "PhD": 80000}
base = np.array([edu_base[e] for e in education_levels])
salary = base + years_exp * 2000 + skill_score * 120 + location_idx * 8000 + np.random.normal(0, 5000, n)
salary = salary.clip(15000, 250000)

df = pd.DataFrame({
    "years_experience": years_exp.round(2),
    "education": education_levels,
    "skill_score": skill_score.round(2),
    "age": age.round(0).astype(int),
    "location_idx": location_idx,
    "salary": salary.round(2)
})

# -----------------------------
# Preprocess + train model
# -----------------------------
X = df.drop(columns=["salary"])
y = df["salary"]

numeric_features = ["years_experience", "skill_score", "age", "location_idx"]
categorical_features = ["education"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features)
])

X_transformed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💰 Salary Prediction App")
st.write("This app predicts **Salary** based on experience, education, skills, age, and location.")

st.sidebar.header("Input Features")
years_experience = st.sidebar.number_input("Years of Experience", 0.0, 50.0, 5.0, step=0.5)
education = st.sidebar.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
skill_score = st.sidebar.slider("Skill Score (0-100)", 0.0, 100.0, 50.0)
age = st.sidebar.number_input("Age", 18, 80, 30, step=1)
location_idx = st.sidebar.selectbox("Location (0=Low cost, 1=Mid, 2=High)", [0, 1, 2])

if st.button("Predict Salary"):
    input_data = pd.DataFrame([{
        "years_experience": years_experience,
        "education": education,
        "skill_score": skill_score,
        "age": age,
        "location_idx": location_idx
    }])
    X_input = preprocessor.transform(input_data)
    pred = model.predict(X_input)[0]
    st.success(f"Predicted Salary: **${pred:,.2f}**")
