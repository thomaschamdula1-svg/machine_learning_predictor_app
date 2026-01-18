#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# advanced_ml_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# -----------------------------
# Page config & dark/light toggle
# -----------------------------
st.set_page_config(
    page_title="Advanced ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar - dark/light mode
mode = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
if mode == "Dark":
    st.markdown(
        """
        <style>
        body {background-color:#222222;color:#ffffff;}
        .stButton>button {background-color:#555555;color:#fff;}
        </style>
        """, unsafe_allow_html=True
    )

# -----------------------------
# Sidebar - Dataset & settings
# -----------------------------
st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
st.sidebar.markdown("---")

algorithm_options = [
    "Linear Regression", "Logistic Regression", "Decision Tree",
    "Random Forest", "KNN", "XGBoost", "Neural Network"
]

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset loaded!")

    target = st.sidebar.selectbox("Select Target variable", df.columns)
    features = st.sidebar.multiselect(
        "Select feature variable", [col for col in df.columns if col != target]
    )

    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)
    algorithm = st.sidebar.selectbox("Select Algorithm", algorithm_options)

    # Encode categorical features automatically
    for col in features:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )

    # -----------------------------
    # Multi-page UI using tabs
    # -----------------------------
    tabs = st.tabs(["Dataset Preview", "Model Training", "Feature Importance", "Download Predictions"])

    # -----------------------------
    # Tab 1: Dataset Preview
    # -----------------------------
    with tabs[0]:
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(10))
        st.markdown(f"**Shape:** {df.shape}")
        st.markdown("**Column Types:**")
        st.write(df.dtypes)
        st.markdown("**Summary Statistics:**")
        st.write(df.describe())

    # -----------------------------
    # Tab 2: Model Training
    # -----------------------------
    with tabs[1]:
        st.markdown(f"### Training {algorithm}")
        if st.button("Run Model"):
            model = None
            preds = None

            # Linear Regression
            if algorithm == "Linear Regression":
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                st.success(f"Mean Squared Error: {mse:.4f}")
                fig = px.scatter(x=y_test, y=preds, labels={"x":"Actual","y":"Predicted"}, title="Actual vs Predicted")
                st.plotly_chart(fig, use_container_width=True)

            # Logistic Regression
            elif algorithm == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=500)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.success(f"Accuracy: {acc:.4f}")
                fig = px.histogram(preds, title="Predicted Class Distribution")
                st.plotly_chart(fig, use_container_width=True)

            # Decision Tree
            elif algorithm == "Decision Tree":
                from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                if y.dtype == 'object' or len(y.unique()) <= 10:
                    model = DecisionTreeClassifier()
                else:
                    model = DecisionTreeRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.success(f"Model trained!")

            # Random Forest
            elif algorithm == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                if y.dtype == 'object' or len(y.unique()) <= 10:
                    model = RandomForestClassifier()
                else:
                    model = RandomForestRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.success(f"Model trained!")

            # KNN
            elif algorithm == "KNN":
                from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
                if y.dtype == 'object' or len(y.unique()) <= 10:
                    model = KNeighborsClassifier()
                else:
                    model = KNeighborsRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.success(f"Model trained!")

            # Neural Network
            elif algorithm == "Neural Network":
                from sklearn.neural_network import MLPClassifier, MLPRegressor
                if y.dtype == 'object' or len(y.unique()) <= 10:
                    model = MLPClassifier(max_iter=500)
                else:
                    model = MLPRegressor(max_iter=500)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.success(f"Model trained!")

            st.session_state['model'] = model
            st.session_state['preds'] = preds
            st.session_state['X_test'] = X_test

    # -----------------------------
    # Tab 3: Feature Importance
    # -----------------------------
    with tabs[2]:
        st.markdown("### Feature Importance / Coefficients")
        if 'model' in st.session_state:
            model = st.session_state['model']
            explainer = None

            if hasattr(model, "feature_importances_"):
                importance = pd.DataFrame({
                    "Feature": features,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                st.dataframe(importance)
                fig = px.bar(importance, x="Feature", y="Importance", title="Feature Importance")
                st.plotly_chart(fig, use_container_width=True)

            elif hasattr(model, "coef_"):
                coef = model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]
                importance = pd.DataFrame({
                    "Feature": features,
                    "Coefficient": coef
                }).sort_values(by="Coefficient", key=abs, ascending=False)
                st.dataframe(importance)
                fig = px.bar(importance, x="Feature", y="Coefficient", title="Feature Coefficients")
                st.plotly_chart(fig, use_container_width=True)

            # SHAP explanation
            try:
                st.markdown("### SHAP Summary Plot")
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)
                shap.summary_plot(shap_values, X_test, show=False)
                st.pyplot(bbox_inches='tight')
            except:
                st.info("SHAP not supported for this model type yet.")

    # -----------------------------
    # Tab 4: Download Predictions
    # -----------------------------
    with tabs[3]:
        st.markdown("### Download Predictions")
        if 'preds' in st.session_state and 'X_test' in st.session_state:
            result_df = st.session_state['X_test'].copy()
            result_df['Prediction'] = st.session_state['preds']
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("Run a model first to generate predictions.")

else:
    st.info("Upload a CSV file to get started.")

