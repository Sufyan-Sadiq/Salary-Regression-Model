# train_ann.py
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load dataset (expects columns: years_experience, education, skill_score, age, location_idx, salary)
df = pd.read_csv("synthetic_salary.csv")  # or your real file

X = df.drop(columns=["salary"])
y = df["salary"].values

numeric_features = ["years_experience","skill_score","age","location_idx"]
categorical_features = ["education"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(sparse=False, handle_unknown="ignore"), categorical_features)
])

X_tr = preprocessor.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_tr, y, test_size=0.2, random_state=42)

# Build ANN
input_dim = X_train.shape[1]
model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=[keras.metrics.RootMeanSquaredError()])

# Train
callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32, callbacks=callbacks)

# Evaluate
y_pred = model.predict(X_val).flatten()
rmse = sqrt(mean_squared_error(y_val, y_pred))
print("Validation RMSE:", rmse)

# Save model and preprocessor
model.save("salary_model.h5")
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("Saved salary_model.h5 and preprocessor.pkl")
