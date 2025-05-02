import streamlit as st
import math
import pathlib, numpy as np, pandas as pd, plotly.express as px, shap, matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="FIES Dashboard", layout="wide")

try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except ImportError:
    CATBOOST_OK = False


DATA_PATH = pathlib.Path("family_income_and_expenditure.csv")

@st.cache_data
def load_data(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["Food_Share"] = (
        df["Total_Food_Expenditure"] / df["Total_Household_Income"].replace({0: np.nan})
    ).clip(0, 1)
    return df

df_base = load_data(DATA_PATH)

POVERTY_LINE = {
    'CAR': 31774,
    'Caraga': 31774,
    'VI - Western Visayas': 33122,
    'V - Bicol Region': 32000,
    'ARMM': 30000,
    'III - Central Luzon': 34000,
    'II - Cagayan Valley': 32500,
    'IVA - CALABARZON': 37097,
    'VII - Central Visayas': 35064,
    'X - Northern Mindanao': 33000,
    'XI - Davao Region': 30861,
    'VIII - Eastern Visayas': 31500,
    'I - Ilocos Region': 33454,
    'NCR': 37000,
    'IVB - MIMAROPA': 32000,
    'XII - SOCCSKSARGEN': 31000,
    'IX - Zamboanga Peninsula': 31000
    }  # extend table



st.title("🇵🇭 FIES 2021 Dashboard – Descriptive & Predictive Analytics")
tab_desc, tab_pred = st.tabs(["Descriptive analytics", "Predictive model"])


with tab_desc:
    st.sidebar.header("Filters")

    # FILTERS
    region_opt = ["All"] + sorted(df_base.Region.unique())
    pick_region = st.sidebar.selectbox("Region", region_opt)

    hh_type_opt = ["All"] + sorted(df_base.Type_of_Household.unique())
    pick_hh_type = st.sidebar.selectbox("Type of Household", hh_type_opt)

    tenure_opt = ["All"] + sorted(df_base.Tenure_Status.unique())
    pick_tenure = st.sidebar.selectbox("Tenure Status", tenure_opt)

    bldg_opt = ["All"] + sorted(df_base.Type_of_Building_or_House.unique())
    pick_bldg = st.sidebar.selectbox("Type of Building or House", bldg_opt)

    job_ind_opt = ["All"] + sorted(df_base.Household_Head_Job_or_Business_Indicator.unique())
    pick_job_ind = st.sidebar.selectbox("Head: Job/Business Indicator", job_ind_opt)

    # Apply filters
    df = df_base.copy()
    if pick_region != "All":
        df = df.query("Region == @pick_region")
    if pick_hh_type != "All":
        df = df.query("Type_of_Household == @pick_hh_type")
    if pick_tenure != "All":
        df = df.query("Tenure_Status == @pick_tenure")
    if pick_bldg != "All":
        df = df.query("Type_of_Building_or_House == @pick_bldg")
    if pick_job_ind != "All":
        df = df.query("Household_Head_Job_or_Business_Indicator == @pick_job_ind")

    k1, k2, k3 = st.columns(3)
    avg_age = df.Household_Head_Age.mean()
    k1.metric("Avg. Head Age", f"{math.floor(avg_age)} yrs")  

    mean_income = df.Total_Household_Income.mean()
    k2.metric("Mean Income (₱)", f"{mean_income:,.0f}")

    median_food_share = df.Food_Share.median()
    k3.metric("Median Food Share", f"{median_food_share:.2%}")

    k4,k5,k6 = st.columns(3)
    k4.metric("Households", f"{len(df):,}")

    average_hh_size = df['Total_Number_of_Family_members'].mean()
    k5.metric("Avg HH Size", f"{math.floor(average_hh_size)}")

    avg_employed = df.Total_number_of_family_members_employed.mean()
    k6.metric("Avg. Fam. Employed", f"{math.floor(avg_employed)}")

    st.markdown("---")

    with st.expander("👁️ Preview data"):
        st.dataframe(df.head(50), use_container_width=True)

    # Treemap of Expenditure Breakdown
    expend_cols = {
        "Total_Food_Expenditure": "Food",
        "Bread_and_Cereals_Expenditure": "Bread and Cereals",
        "Total_Rice_Expenditure": "Rice",
        "Meat_Expenditure": "Meat",
        "Total_Fish_and__marine_products_Expenditure": "Fish",
        "Fruit_Expenditure": "Fruit",
        "Vegetables_Expenditure": "Vegetables",
        "Restaurant_and_hotels_Expenditure": "Restaurants and Hotels",
        "Alcoholic_Beverages_Expenditure": "Aloholic Beverages",
        "Tobacco_Expenditure": "Tobacco",
        "Clothing,_Footwear_and_Other_Wear_Expenditure": "Clothing",
        "Housing_and_water_Expenditure": "Housing and Water",
        "Medical_Care_Expenditure": "Medical Care",
        "Transportation_Expenditure": "Transport",
        "Communication_Expenditure": "Communication",
        "Miscellaneous_Goods_and_Services_Expenditure": "Miscellaneous Goods and Services",
        "Special_Occasions_Expenditure": "Special Occasions",
        "Crop_Farming_and_Gardening_expenses": "Crop and Farming",

    }

    treemap_df = (
        df[list(expend_cols.keys())]
        .sum()
        .rename_axis("Category")
        .reset_index(name="PHP")
        .replace({"Category": expend_cols})
    )

    fig_tm = px.treemap(
        treemap_df,
        path=["Category"],
        values="PHP",
        title="Expenditure Breakdown",
    )
    fig_tm.update_traces(
      hovertemplate="<b>%{label}</b><br>%{percentEntry:.1%} of total<extra></extra>",
    )
    st.plotly_chart(fig_tm, use_container_width=True)

    # Pie Chart Main Sources of Income
    pie_df = (
        df["Main_Source_of_Income"]
        .value_counts()
        .rename_axis("Source")
        .reset_index(name="Households")
    )

    fig_income = px.pie(
        pie_df,
        names="Source",
        values="Households",
        title="Main Source of Income",
        hole=0,
    )

    fig_income.update_traces(
        textinfo="label+percent",
        pull=[0.05 if v == pie_df.Households.max() else 0 for v in pie_df.Households]
    )

    st.plotly_chart(fig_income, use_container_width=True)

    (df["Main_Source_of_Income"].value_counts().rename_axis("Source").reset_index(name="Households"))


    # Bar Chart of Head-of-Household Education Level
    EDU_COL = "Household_Head_Highest_Grade_Completed"
    THRESH  = 0.01 # group categories that are <1 % of households

    freq = df[EDU_COL].value_counts(dropna=False)
    small = (freq / freq.sum()) < THRESH

    major = (
        freq[~small]
        .rename_axis("Education")
        .reset_index(name="Households")
    )

    if small.any():
        other_total = freq[small].sum()
        major = pd.concat(
            [major, pd.DataFrame({"Education": ["Other (<1 % each)"], "Households": [other_total]})],
            ignore_index=True,
        )

    # sort low → high so bars read bottom-to-top
    major = major.sort_values("Households", ascending=True)

    fig4 = px.bar(
        major,
        x="Households",
        y="Education",
        orientation="h",
        text="Households",
        title="Head-of-Household Education Level",
    )

    fig4.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig4.update_layout(
        yaxis_title="",
        xaxis_title="Number of households",
        margin=dict(l=110, r=20, t=60, b=40),
    )

    st.plotly_chart(fig4, use_container_width=True)

    # Breakdown of 'Other' categories
    if small.any():
        other_break = (
            freq[small]
            .rename_axis("Education")
            .reset_index(name="Households")
            .sort_values("Households", ascending=False)
        )
        with st.expander("See breakdown of 'Other' categories"):
            st.dataframe(other_break, use_container_width=True)

# Predictive tab
with tab_pred:
    st.header("🏷️ Poverty-Risk Classifier")

    # 1 ── Build modelling frame
    @st.cache_data
    def model_df(src: pd.DataFrame) -> pd.DataFrame:
        df = src.copy()
        df["poverty_line"]  = df.Region.map(POVERTY_LINE)
        df["PerCap_Income"] = df.Total_Household_Income / df.Total_Number_of_Family_members.clip(lower=1)
        df["is_poor"]       = (df.PerCap_Income < df.poverty_line).astype(int)
        return df

    df_mod = model_df(df_base)

    # feature lists
    cat_cols = [
        "Region",
        "Main_Source_of_Income",
        "Household_Head_Sex",
        "Household_Head_Marital_Status",
        "Household_Head_Highest_Grade_Completed",
        "Household_Head_Job_or_Business_Indicator",
        "Household_Head_Occupation",
        "Household_Head_Class_of_Worker",
        "Type_of_Household",
        "Tenure_Status",
        "Electricity",
        "Type_of_Building_or_House",
        "Type_of_Roof",
        "Type_of_Walls",
        "Toilet_Facilities",
        "Main_Source_of_Water_Supply"
    ]
    num_cols = [
        "Total_Household_Income",
        "Total_Food_Expenditure",
        "Food_Share",
        "Agricultural_Household_indicator",
        "Bread_and_Cereals_Expenditure",
        "Total_Rice_Expenditure",
        "Meat_Expenditure",
        "Total_Fish_and__marine_products_Expenditure",
        "Fruit_Expenditure",
        "Vegetables_Expenditure",
        "Restaurant_and_hotels_Expenditure",
        "Alcoholic_Beverages_Expenditure",
        "Tobacco_Expenditure",
        "Clothing,_Footwear_and_Other_Wear_Expenditure",
        "Housing_and_water_Expenditure",
        "Imputed_House_Rental_Value",
        "Medical_Care_Expenditure",
        "Transportation_Expenditure",
        "Communication_Expenditure",
        "Education_Expenditure",
        "Miscellaneous_Goods_and_Services_Expenditure",
        "Special_Occasions_Expenditure",
        "Crop_Farming_and_Gardening_expenses",
        "Total_Income_from_Entrepreneurial_Acitivites",
        "Household_Head_Age",
        "Total_Number_of_Family_members",
        "Members_with_age_less_than_5_year_old",
        "Members_with_age_5_-_17_years_old",
        "Total_number_of_family_members_employed",
        "House_Floor_Area",
        "House_Age",
        "Number_of_bedrooms",
        "Number_of_Television",
        "Number_of_CD/VCD/DVD",
        "Number_of_Component/Stereo_set",
        "Number_of_Refrigerator/Freezer",
        "Number_of_Washing_Machine",
        "Number_of_Airconditioner",
        "Number_of_Car,_Jeep,_Van",
        "Number_of_Landline/wireless_telephones",
        "Number_of_Cellular_phone",
        "Number_of_Personal_Computer",
        "Number_of_Stove_with_Oven/Gas_Range",
        "Number_of_Motorized_Banca",
        "Number_of_Motorcycle/Tricycle",
    ]

    X, y = df_mod[cat_cols + num_cols], df_mod["is_poor"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2 ── Model selector (CatBoost = bonus)
    algo_list = ["Logistic Regression (Default)"] + (["CatBoost (Recommended)"] if CATBOOST_OK else [])
    algo = st.radio("Choose algorithm", algo_list)

    # 3 ── Build pipeline / estimator
    def make_estimator(name: str):
        if name.startswith("CatBoost"):
            cat_idx = [X.columns.get_loc(c) for c in cat_cols]
            return CatBoostClassifier(
                depth=6, learning_rate=0.1, iterations=500,
                auto_class_weights="Balanced", verbose=False,
                random_state=42, cat_features=cat_idx
            )
        pre   = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                                  remainder="passthrough")
        model = LogisticRegression(max_iter=200, class_weight="balanced") \
                if name == "Logistic Regression (Default)" \
                else DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
        return Pipeline([("pre", pre), ("model", model)])

    # 4 ── Train & evaluate
    if st.button("🚀 Train model"):
        clf = make_estimator(algo)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # store for inference
        st.session_state["fitted_clf"] = clf
        st.session_state["algo"]      = algo

        # metrics
        roc  = roc_auc_score(y_test, y_prob)
        prc  = average_precision_score(y_test, y_prob)

        # lift @ top-20 %
        top20 = int(0.2 * len(y_prob))
        idx   = np.argsort(y_prob)[::-1][:top20]
        lift  = y_test.iloc[idx].mean() / y.mean()

        st.success(f"ROC-AUC {roc:.3f}  PR-AUC {prc:.3f}  Lift@20 % {lift:.2f}×")

        cm = confusion_matrix(y_test, (y_prob > 0.5).astype(int))
        st.write(pd.DataFrame(cm,
                 index=["Actual 0","Actual 1"],
                 columns=["Pred 0","Pred 1"]))

        # 4a ── SHAP explanation
        with st.expander("🔍 SHAP explanation"):
            if algo.startswith("CatBoost"):
                explainer = shap.TreeExplainer(clf)
                shap_vals = explainer.shap_values(X_test)
                base_df   = X_test
            elif algo == "Logistic Regression (Default)":
                X_pre = clf["pre"].transform(X_test)
                if hasattr(X_pre, "toarray"):       # convert sparse → dense
                    X_pre = X_pre.toarray()

                feature_names = clf["pre"].get_feature_names_out()
                base_df = pd.DataFrame(X_pre, columns=feature_names)   # << give names

                explainer = shap.Explainer(clf["model"], base_df)      # << pass DataFrame
                shap_vals = explainer(base_df)
            fig,_ = plt.subplots()
            shap.summary_plot(shap_vals, base_df, show=False)
            st.pyplot(fig)

    # 5 ── Single-household scorer
    st.markdown("---")
    st.subheader("🎯 Score a single household")

    with st.form("score_form"):
        income     = st.number_input("Total household income (₱)", 0, 1_000_000, 120_000)
        hh_size    = st.number_input("Household size", 1, 15, 5)
        food_exp   = st.number_input("Annual food expenditure (₱)", 0, 500_000, 60_000)
        region_one = st.selectbox("Region", sorted(df_base.Region.unique()))
        educ       = st.selectbox("Head highest grade", df_base.Household_Head_Highest_Grade_Completed.unique())
        submitted  = st.form_submit_button("Predict")

    if submitted:
        if "fitted_clf" not in st.session_state:
            st.error("⚠️ Train a model first.")
            st.stop()

        sample = pd.DataFrame([{
            "Total_Household_Income": income,
            "Total_Food_Expenditure": food_exp,
            "Food_Share": food_exp / max(income,1),
            "Household_Head_Age": 40,
            "Total_Number_of_Family_members": hh_size,
            "Members_with_age_5_-_17_years_old": 2,
            "Region": region_one,
            "Main_Source_of_Income": "Wages/Salaries",
            "Household_Head_Sex": "Male",
            "Household_Head_Marital_Status": "Married",
            "Household_Head_Highest_Grade_Completed": educ,
            "Type_of_Household": "Single-family",
            "Tenure_Status": "Own_or_amortized",
            "Electricity": "Yes",
        }])

        clf  = st.session_state["fitted_clf"]
        if st.session_state["algo"].startswith("CatBoost"):
            for col in cat_cols:
                sample[col] = sample[col].astype(str)
        prob = clf.predict_proba(sample)[:,1][0]
        st.success(f"Predicted poverty probability: {prob:.2%}")
        st.progress(prob)

st.caption("Data: PSA FIES 2021 • Dashboard built with Streamlit & CatBoost")
