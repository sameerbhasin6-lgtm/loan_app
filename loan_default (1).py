import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="Loan Default Marketing Analytics", layout="wide")

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("loan_default.xlsx")
        return df
    except FileNotFoundError:
        st.error("File 'loan_default.xlsx' not found. Upload the file to the repo or folder.")
        return None

df = load_data()

if df is not None:
    # --- 2. DATA PREPROCESSING ---
    df_model = df.copy()
    le_dict = {}

    for col in df_model.columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le

    X = df_model.drop("Default", axis=1)
    y = df_model["Default"]

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)

    yes_index = list(le_dict["Default"].classes_).index("Yes")
    df["Risk_Probability"] = clf.predict_proba(X)[:, yes_index]

    # --- 3. DASHBOARD ---
    st.title("🏦 Loan Default Marketing Analytics Dashboard")
    st.divider()

    # KPI
    total_defaults = df[df["Default"] == "Yes"].shape[0]
    total_customers = df.shape[0]
    default_rate = (total_defaults / total_customers) * 100

    st.metric(label="Overall Default Rate", value=f"{default_rate:.1f}%")

    st.divider()

    # Charts
    st.subheader("📈 Risk Drivers")

    def get_risk_by_category(column_name):
        risk_df = df.groupby(column_name)["Default"].apply(
            lambda x: (x == "Yes").sum() / len(x) * 100
        ).reset_index()
        risk_df.columns = [column_name, "Default Rate (%)"]
        return risk_df.sort_values("Default Rate (%)", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        emp_risk = get_risk_by_category("Employment_Type")
        fig = px.bar(emp_risk, x="Employment_Type", y="Default Rate (%)",
                     color="Default Rate (%)", color_continuous_scale="Reds",
                     title="Risk by Employment Type")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cred_risk = get_risk_by_category("Credit_History")
        fig = px.bar(cred_risk, x="Credit_History", y="Default Rate (%)",
                     color="Default Rate (%)", color_continuous_scale="Reds",
                     title="Risk by Credit History")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # High risk table
    st.subheader("👤 High-Risk Customer Profiles")

    top_risky = df.sort_values("Risk_Probability", ascending=False).head(5)
    display_cols = ["Employment_Type", "Credit_History", "Income_Bracket",
                    "Education_Level", "Risk_Probability"]

    top_risky["Risk_Probability"] = top_risky["Risk_Probability"].apply(
        lambda x: f"{x*100:.1f}%"
    )

    st.dataframe(top_risky[display_cols], use_container_width=True)

else:
    st.warning("Upload the dataset to view the dashboard.")
