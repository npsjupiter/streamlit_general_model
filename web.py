import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
import joblib

st.set_page_config(
    page_title="Model Selection App",
    layout="wide"
)

st.markdown("""
    <style>
        .main {
            background-color: #fdf6f0;
            background-image: linear-gradient(120deg, #fceabb 0%, #f8b500 100%);
            padding: 20px;
            border-radius: 10px;
        }
        .stApp {
            background-color: #fff0e5;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Smart & Silly Model Selector")
with st.spinner("Getting everything ready for your data science journey..."):
    time.sleep(2)
    st.success("Ready to roll!")
    st.toast("Welcome to the Smart Model Selector")
    # st.balloons()

st.markdown("""
Welcome to the one-stop shop for all your model training adventures!
Here you can:
- Upload a dataset (don't worry, we won’t judge your CSV)
- Clean up messy data (we all have baggage)
- Encode those pesky categories
- Train a model like a champ!
- Evaluate performance with style
""")

# Upload CSV file
df_file = st.file_uploader("Drop your CSV like it's hot", type="csv")
if df_file is not None:
    df = pd.read_csv(df_file)
    st.toast("File received! Let's explore! ")
    st.success("Woohoo! File uploaded successfully!")
    st.write("###  Sneak Peek at Your Data:")
    st.dataframe(df.head())

    with st.expander("Basic Info (a little nosy but helpful)"):
        st.write("Shape:", df.shape)
        st.write("Data Types:")
        st.write(df.dtypes)
        st.write("Numerical Columns:", list(df.select_dtypes(include=[np.number]).columns))
        st.write("Categorical Columns:", list(df.select_dtypes(exclude=[np.number]).columns))

    with st.expander("Missing Value Patrol"):
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        missing_df = missing_df[missing_df["Missing Values"] > 0]
        st.write(missing_df)

        if not missing_df.empty:
            method = st.radio("Pick a Cleanup Method", ["Delete Rows (Ruthless)", "Impute Values (Kind & Caring)"])
            if method == "Delete Rows (Ruthless)":
                df.dropna(inplace=True)
                st.warning("Gone. Forever. Like socks in the dryer. ")
            else:
                num_cols = df.select_dtypes(include=[np.number]).columns
                cat_cols = df.select_dtypes(exclude=[np.number]).columns
                df[num_cols] = df[num_cols].fillna(0)
                df[cat_cols] = df[cat_cols].fillna("Unknown")
                st.success("Like magic — no more missing values!")

    with st.expander("Let's Talk Categorical Drama"):
        cat_col1 = df.select_dtypes(exclude=[np.number]).columns
        cat_col2=st.multiselect("Choose Your Feature Squad",cat_col1)
        cat_cols=df[cat_col2].columns
        if cat_cols.any():
            encoding_method = st.radio("Your Encoding Style", ["One-Hot Encoding (Extra!)", "Label Encoding (Straightforward)"])
            if encoding_method == "One-Hot Encoding (Extra!)":
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            else:
                le = LabelEncoder()
                for col in cat_cols:
                    df[col] = le.fit_transform(df[col])
            st.success("Drama resolved with encoding therapy!")

    st.markdown("---")
    st.header("Model Training Ground")

    target_col = st.selectbox("Choose Your Target Variable (The Chosen One)", df.columns)
    features = st.multiselect("Choose Your Feature Squad", [col for col in df.columns if col != target_col], default=[col for col in df.columns if col != target_col])

    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    problem_type = st.radio("What's the Nature of Your Quest?", ["Regression", "Classification"])

    if problem_type == "Regression":
        model_option = st.selectbox("Pick a Regression Model", ["Linear Regression", "Decision Tree Regressor"])
        if model_option == "Linear Regression":
            model = LinearRegression()
        else:
            depth = st.slider("How Deep Should the Tree Go?", 2, 20, value=5)
            min_samples = st.slider(" Minimum Samples to Split", 2, 100, value=10)
            model = DecisionTreeRegressor(max_depth=depth, min_samples_split=min_samples)

        with st.spinner("Training your regression model..."):
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            st.subheader("Train Set Metrics")
            st.metric("R² Score (Train)", round(float(r2_score(y_train, y_train_pred)), 3))
            st.metric("MAE (Train)", round(float(mean_absolute_error(y_train, y_train_pred)), 3))
            st.metric("RMSE (Train)", round(float(np.sqrt(mean_squared_error(y_train, y_train_pred))), 3))

            st.subheader("Test Set Metrics")
            st.metric("R² Score (Test)", round(float(r2_score(y_test, y_test_pred)), 3))
            st.metric("MAE (Test)", round(float(mean_absolute_error(y_test, y_test_pred)), 3))
            st.metric("RMSE (Test)", round(float(np.sqrt(mean_squared_error(y_test, y_test_pred))), 3))

    else:
        model_option = st.selectbox("Choose a Classification Model", ["Logistic Regression", "Decision Tree Classifier"])
        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            depth = st.slider("Max Tree Depth?", 2, 20, value=5)
            min_samples = st.slider("Min Samples Split", 2, 100, value=10)
            model = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_samples)

        with st.spinner("Training your classification model..."):
            model.fit(X_train, y_train)
            joblib.dump(model, "model.joblib")
            st.write("Model Saved")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            st.subheader("Train Set Metrics")
            st.metric("Accuracy (Train)", round(float(accuracy_score(y_train, y_train_pred)), 3))

            st.subheader("Test Set Metrics")
            st.metric("Accuracy (Test)", round(float(accuracy_score(y_test, y_test_pred)), 3))

            st.subheader(" Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='coolwarm', ax=ax)
            st.pyplot(fig)

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_test_pred))

    # st.snow()
    # st.balloons()
    st.toast("Model training and evaluation complete! You're a Data Hero! ")

    st.markdown("---")
    st.header(" Download Predictions")

    predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_test_pred})
    csv_data = predictions_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_data, file_name="predictions.csv", mime="text/csv")