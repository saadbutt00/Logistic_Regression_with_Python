import streamlit as st
import numpy as np
import pandas as pd

# --- Helper Functions ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_logistic_regression(X, Y, epochs, lr):
    m, num_features = X.shape
    W = np.zeros(num_features)
    b = 0.0

    history = []  # to store w, b, loss over time

    for i in range(epochs):
        z = np.dot(X, W) + b
        y_pred = sigmoid(z)
        loss = compute_loss(Y, y_pred)

        dw = np.dot((y_pred - Y), X) / m
        db = np.sum(y_pred - Y) / m

        W -= lr * dw
        b -= lr * db

        if i % 100 == 0 or i == epochs-1:
            history.append({
                'Epoch': i,
                'Loss': np.round(loss, 4),
                'Bias': np.round(b, 4),
                'Weights': np.round(W, 4)
            })

    return W, b, loss, history

def predict(X, W, b):
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)
    return y_pred

# --- New Suggestion Function ---
def suggest_hyperparams(loss):
    if loss > 0.5:
        return "🔧 High loss: Increase epochs (e.g. +500) and lower learning rate slightly (e.g. 0.01 or 0.05)."
    elif 0.3 < loss <= 0.5:
        return "⚠️ Moderate loss: Try increasing epochs (+200) or reducing learning rate slightly."
    elif 0.15 < loss <= 0.3:
        return "✅ Decent loss: Model is doing well. Slight fine-tuning may help."
    else:
        return "🎯 Excellent loss: No further tuning likely needed."

# --- Streamlit App ---
st.set_page_config(page_title="Logistic Regression", layout="centered")

st.title("Logistic Regression Model")

# Definition
st.write("Logistic Regression (Sigmoid) is a classification algorithm used to predict binary outcomes using probabilities.")
st.write("**Example:** Predict if a student passes or fails based on study hours.")

# Instructions
st.markdown("**Here, how you can predict your values:**")
st.markdown("""
**Step 1**: Enter X's values - You can also name X columns as you want.

**Step 2**: Enter Y values - Number of values in Y should match the number of values in X.

**Step 3**: Set epochs & learning rate.

**Step 4**: Click 'Submit' to train the model.
            
**Step 5**: Enter values for prediction & get help from suggestion.
""")

# Inputs
num_features = st.number_input("Enter Number of Features:", min_value=1, step=1, value=1)

feature_names = []
feature_data = []

# Feature inputs
for i in range(num_features):
    fname = st.text_input(f"Enter name for Feature {i+1}:", value=f"X{i+1}")
    feature_names.append(fname)
    values = st.text_input(f"Enter values for {fname} (space separated):", key=f"values_{i}")
    if values:
        feature_data.append(list(map(float, values.strip().split())))

# Target feature name
y_feature_name = st.text_input("Enter name for Target Feature (Y):", value="Result")

# Label names
y_name0 = st.text_input("Name for Class 0:")
y_name1 = st.text_input("Name for Class 1:")

# Target values
y_values = st.text_input(f"Enter {y_feature_name} values (space separated, 0 or 1):")

# Training parameters
epochs = st.number_input("Number of Epochs:", min_value=10, value=1000, step=10)
lr = st.number_input("Learning Rate:", min_value=0.001, value=0.1, step=0.01)

# Session state to store model parameters after training
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Train button
if st.button("Submit"):
    if len(feature_data) != num_features:
        st.error("Please enter all feature values.")
    else:
        try:
            X = np.array(feature_data).T
            m = X.shape[0]
            Y = np.array(list(map(int, y_values.strip().split())))

            if len(Y) != m:
                st.error("Y values count must match number of samples.")
            else:
                W, b, final_loss, history = train_logistic_regression(X, Y, epochs, lr)
                st.session_state.W = W
                st.session_state.b = b
                st.session_state.final_loss = final_loss
                st.session_state.history = history
                st.session_state.model_trained = True
                st.success("Model Trained")

                # Display table with Bias & Loss added
                df_history = pd.DataFrame(history)
                st.write("Training Progress:")
                st.dataframe(df_history)

                # Combine Weights, Bias and Loss into a single dataframe
                result_data = {
                    'Weights': [np.round(W, 4)],
                    'Bias': [np.round(b, 4)],
                    'Loss': [np.round(final_loss, 4)]
                }
                result_df = pd.DataFrame(result_data)
                st.write("### 🔎 Final Model Parameters")
                st.dataframe(result_df)

        except Exception as e:
            st.error("Invalid input. Check your values.")

# Prediction block: always visible if model trained
if st.session_state.model_trained:
    st.subheader("🔮 Make Predictions")

    input_data = []
    for fname in feature_names:
        val = st.number_input(f"Enter value for {fname}:", key=f"pred_{fname}")
        input_data.append(val)

    if st.button("Predict"):
        input_data = np.array(input_data).reshape(1, -1)
        y_pred = predict(input_data, st.session_state.W, st.session_state.b)
        y_pred_label = int(y_pred >= 0.5)

        st.write("**Predicted Probability:**", np.round(float(y_pred), 4))
        st.write("**Predicted Class:**", f"{y_name1}" if y_pred_label == 1 else f"{y_name0}")

        st.subheader("📉 Loss Function Value:")
        st.write("The loss value shows how well model fits the data.")
        st.write("**Loss:**", np.round(st.session_state.final_loss, 4))

        st.subheader("🧠 Suggestion:")
        suggestion = suggest_hyperparams(st.session_state.final_loss)
        st.write(suggestion)

# streamlit run "f:/ML_with_Python/Logistic Regression/app.py"