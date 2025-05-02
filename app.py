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

# https://psa.gov.ph/system/files/phdsd/Highlights%20of%20the%202023%201st%20sem%20Official%20Poverty%20Statistics.pdf
# 2018 data
POVERTY_LINE = {
    'CAR': 12358*12,
    'Caraga': 12346*12,
    'VI - Western Visayas': 11964*12,
    'V - Bicol Region': 11975*12,
    'ARMM': 13599*12,
    'III - Central Luzon': 12976*12,
    'II - Cagayan Valley': 12182*12,
    'IVA - CALABARZON': 13669*12,
    'VII - Central Visayas': 12724*12,
    'X - Northern Mindanao': 12259*12,
    'XI - Davao Region': 12718*12,
    'VIII - Eastern Visayas': 12195*12,
    'I - Ilocos Region': 12837*12,
    'NCR': 14102*12,
    'IVB - MIMAROPA': 11472*12,
    'XII - SOCCSKSARGEN': 12082*12,
    'IX - Zamboanga Peninsula': 12424*12
    } 



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
                if hasattr(X_pre, "toarray"):
                    X_pre = X_pre.toarray()

                feature_names = clf["pre"].get_feature_names_out()
                base_df = pd.DataFrame(X_pre, columns=feature_names)

                explainer = shap.Explainer(clf["model"], base_df)
                shap_vals = explainer(base_df)
            fig,_ = plt.subplots()
            shap.summary_plot(shap_vals, base_df, show=False)
            st.pyplot(fig)

    # 5 ── Single-household scorer
    st.markdown("---")
    st.subheader("🎯 Score a single household")

    with st.form("score_form"):

        region_one = st.selectbox(
            "Region",
            sorted(df_base["Region"].unique())
        )

        main_income = st.selectbox(
            "Main source of income",
            df_base["Main_Source_of_Income"].unique()
        )

        head_sex = st.selectbox(
            "Household head sex",
            df_base["Household_Head_Sex"].unique()
        )

        marital = st.selectbox(
            "Head marital status",
            df_base["Household_Head_Marital_Status"].unique()
        )

        education = st.selectbox(
            "Head highest grade completed",
            df_base["Household_Head_Highest_Grade_Completed"].unique()
        )

        job_indicator = st.selectbox(
            "Head job/business indicator",
            df_base["Household_Head_Job_or_Business_Indicator"].unique()
        )

        occupation = st.selectbox(
            "Head occupation",
            df_base["Household_Head_Occupation"].unique()
        )

        class_worker = st.selectbox(
            "Head class of worker",
            df_base["Household_Head_Class_of_Worker"].unique()
        )

        hh_type = st.selectbox(
            "Type of household",
            df_base["Type_of_Household"].unique()
        )

        tenure = st.selectbox(
            "Tenure status",
            df_base["Tenure_Status"].unique()
        )

        electricity = st.selectbox(
            "Electricity",
            df_base["Electricity"].unique()
        )

        building = st.selectbox(
            "Type of building/house",
            df_base["Type_of_Building_or_House"].unique()
        )

        roof = st.selectbox(
            "Type of roof",
            df_base["Type_of_Roof"].unique()
        )

        walls = st.selectbox(
            "Type of walls",
            df_base["Type_of_Walls"].unique()
        )

        toilet = st.selectbox(
            "Toilet facilities",
            df_base["Toilet_Facilities"].unique()
        )

        water = st.selectbox(
            "Main source of water supply",
            df_base["Main_Source_of_Water_Supply"].unique()
        )

        ## Numeric inputs

        # 1) Total household income
        income = st.number_input(
            "Total household income (₱)",
            min_value=0,
            max_value=1_000_000,
            value=int(df_base["Total_Household_Income"].median())
        )

        # 2) Total food expenditure
        food_exp = st.number_input(
            "Total food expenditure (₱)",
            min_value=0,
            max_value=500_000,
            value=int(df_base["Total_Food_Expenditure"].median())
        )

        # 3) Food share
        food_share = st.number_input(
            "Food share (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(df_base["Food_Share"].median())
        )

        # 4) Agricultural household indicator
        agri_hh = st.number_input(
            "Agricultural household indicator",
            min_value=0,
            max_value=1,
            value=int(df_base["Agricultural_Household_indicator"].mode()[0])
        )

        # 5) Bread & cereals expenditure
        bread_cereals = st.number_input(
            "Bread & cereals expenditure (₱)",
            min_value=0,
            max_value=200_000,
            value=int(df_base["Bread_and_Cereals_Expenditure"].median())
        )

        # 6) Total rice expenditure
        rice_exp = st.number_input(
            "Total rice expenditure (₱)",
            min_value=0,
            max_value=200_000,
            value=int(df_base["Total_Rice_Expenditure"].median())
        )

        # 7) Meat expenditure
        meat_exp = st.number_input(
            "Meat expenditure (₱)",
            min_value=0,
            max_value=200_000,
            value=int(df_base["Meat_Expenditure"].median())
        )

        # 8) Fish & marine products expenditure
        fish_exp = st.number_input(
            "Total fish & marine products expenditure (₱)",
            min_value=0,
            max_value=200_000,
            value=int(df_base["Total_Fish_and__marine_products_Expenditure"].median())
        )

        # 9) Fruit expenditure
        fruit_exp = st.number_input(
            "Fruit expenditure (₱)",
            min_value=0,
            max_value=100_000,
            value=int(df_base["Fruit_Expenditure"].median())
        )

        # 10) Vegetables expenditure
        veg_exp = st.number_input(
            "Vegetables expenditure (₱)",
            min_value=0,
            max_value=100_000,
            value=int(df_base["Vegetables_Expenditure"].median())
        )

        # 11) Restaurant & hotels expenditure
        resto_exp = st.number_input(
            "Restaurant & hotels expenditure (₱)",
            min_value=0,
            max_value=100_000,
            value=int(df_base["Restaurant_and_hotels_Expenditure"].median())
        )

        # 12) Alcoholic beverages expenditure
        alc_exp = st.number_input(
            "Alcoholic beverages expenditure (₱)",
            min_value=0,
            max_value=100_000,
            value=int(df_base["Alcoholic_Beverages_Expenditure"].median())
        )

        # 13) Tobacco expenditure
        tob_exp = st.number_input(
            "Tobacco expenditure (₱)",
            min_value=0,
            max_value=100_000,
            value=int(df_base["Tobacco_Expenditure"].median())
        )

        # 14) Clothing, footwear & other wear expenditure
        cloth_exp = st.number_input(
            "Clothing, footwear & other wear expenditure (₱)",
            min_value=0,
            max_value=100_000,
            value=int(df_base["Clothing,_Footwear_and_Other_Wear_Expenditure"].median())
        )

        # 15) Housing & water expenditure
        house_water_exp = st.number_input(
            "Housing & water expenditure (₱)",
            min_value=0,
            max_value=200_000,
            value=int(df_base["Housing_and_water_Expenditure"].median())
        )

        # 16) Imputed house rental value
        imputed_rent = st.number_input(
            "Imputed house rental value (₱)",
            min_value=0,
            max_value=200_000,
            value=int(df_base["Imputed_House_Rental_Value"].median())
        )

        # 17) Medical care expenditure
        med_exp = st.number_input(
            "Medical care expenditure (₱)",
            min_value=0,
            max_value=100_000,
            value=int(df_base["Medical_Care_Expenditure"].median())
        )

        # 18) Transportation expenditure
        trans_exp = st.number_input(
            "Transportation expenditure (₱)",
            min_value=0,
            max_value=100_000,
            value=int(df_base["Transportation_Expenditure"].median())
        )

        # 19) Communication expenditure
        comm_exp = st.number_input(
            "Communication expenditure (₱)",
            min_value=0,
            max_value=50_000,
            value=int(df_base["Communication_Expenditure"].median())
        )

        # 20) Education expenditure
        edu_exp = st.number_input(
            "Education expenditure (₱)",
            min_value=0,
            max_value=100_000,
            value=int(df_base["Education_Expenditure"].median())
        )

        # 21) Miscellaneous goods & services expenditure
        misc_exp = st.number_input(
            "Miscellaneous goods & services expenditure (₱)",
            min_value=0,
            max_value=100_000,
            value=int(df_base["Miscellaneous_Goods_and_Services_Expenditure"].median())
        )

        # 22) Special occasions expenditure
        special_exp = st.number_input(
            "Special occasions expenditure (₱)",
            min_value=0,
            max_value=50_000,
            value=int(df_base["Special_Occasions_Expenditure"].median())
        )

        # 23) Crop farming & gardening expenses
        crop_exp = st.number_input(
            "Crop farming & gardening expenses (₱)",
            min_value=0,
            max_value=50_000,
            value=int(df_base["Crop_Farming_and_Gardening_expenses"].median())
        )

        # 24) Total income from entrepreneurial activities
        entrepreneurial_inc = st.number_input(
            "Total income from entrepreneurial activities (₱)",
            min_value=0,
            max_value=1_000_000,
            value=int(df_base["Total_Income_from_Entrepreneurial_Acitivites"].median())
        )

        # 25) Household head age
        head_age = st.number_input(
            "Household head age",
            min_value=0,
            max_value=120,
            value=int(df_base["Household_Head_Age"].median())
        )

        # 26) Total number of family members
        hh_size = st.number_input(
            "Total number of family members",
            min_value=1,
            max_value=20,
            value=int(df_base["Total_Number_of_Family_members"].median())
        )

        # 27) Members aged <5
        under5 = st.number_input(
            "Members aged less than 5 years old",
            min_value=0,
            max_value=10,
            value=int(df_base["Members_with_age_less_than_5_year_old"].median())
        )

        # 28) Members aged 5–17
        age5_17 = st.number_input(
            "Members aged 5–17 years old",
            min_value=0,
            max_value=20,
            value=int(df_base["Members_with_age_5_-_17_years_old"].median())
        )

        # 29) Employed family members
        employed_members = st.number_input(
            "Total number of employed family members",
            min_value=0,
            max_value=20,
            value=int(df_base["Total_number_of_family_members_employed"].median())
        )

        # 30) House floor area
        floor_area = st.number_input(
            "House floor area (sqm)",
            min_value=0,
            max_value=500,
            value=int(df_base["House_Floor_Area"].median())
        )

        # 31) House age
        house_age = st.number_input(
            "House age (years)",
            min_value=0,
            max_value=100,
            value=int(df_base["House_Age"].median())
        )

        # 32) Number of bedrooms
        bedrooms = st.number_input(
            "Number of bedrooms",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_bedrooms"].median())
        )

        # 33) Number of television
        tv_count = st.number_input(
            "Number of televisions",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_Television"].median())
        )

        # 34) Number of CD/VCD/DVD players
        dvd_count = st.number_input(
            "Number of CD/VCD/DVD players",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_CD/VCD/DVD"].median())
        )

        # 35) Number of stereo sets
        stereo_count = st.number_input(
            "Number of component/stereo sets",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_Component/Stereo_set"].median())
        )

        # 36) Number of refrigerator/freezers
        fridge_count = st.number_input(
            "Number of refrigerator/freezers",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_Refrigerator/Freezer"].median())
        )

        # 37) Number of washing machines
        washer_count = st.number_input(
            "Number of washing machines",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_Washing_Machine"].median())
        )

        # 38) Number of air conditioners
        ac_count = st.number_input(
            "Number of air conditioners",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_Airconditioner"].median())
        )

        # 39) Number of cars/jeeps/vans
        car_count = st.number_input(
            "Number of cars/jeeps/vans",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_Car,_Jeep,_Van"].median())
        )

        # 40) Number of landline/wireless telephones
        phone_landline = st.number_input(
            "Number of landline/wireless telephones",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_Landline/wireless_telephones"].median())
        )

        # 41) Number of cellular phones
        phone_cell = st.number_input(
            "Number of cellular phones",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_Cellular_phone"].median())
        )

        # 42) Number of personal computers
        pc_count = st.number_input(
            "Number of personal computers",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_Personal_Computer"].median())
        )

        # 43) Number of stoves with oven/gas range
        stove_count = st.number_input(
            "Number of stoves with oven/gas range",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_Stove_with_Oven/Gas_Range"].median())
        )

        # 44) Number of motorized bancas
        banca_count = st.number_input(
            "Number of motorized bancas",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_Motorized_Banca"].median())
        )

        # 45) Number of motorcycles/tricycles
        motorcycle_count = st.number_input(
            "Number of motorcycles/tricycles",
            min_value=0,
            max_value=10,
            value=int(df_base["Number_of_Motorcycle/Tricycle"].median())
        )

        submitted  = st.form_submit_button("Predict")

    if submitted:
        if "fitted_clf" not in st.session_state:
            st.error("⚠️ Train a model first.")
            st.stop()

        sample = pd.DataFrame([{
            # categorical
            "Region": region_one,
            "Main_Source_of_Income": main_income,
            "Household_Head_Sex": head_sex,
            "Household_Head_Marital_Status": marital,
            "Household_Head_Highest_Grade_Completed": education,
            "Household_Head_Job_or_Business_Indicator": job_indicator,
            "Household_Head_Occupation": occupation,
            "Household_Head_Class_of_Worker": class_worker,
            "Type_of_Household": hh_type,
            "Tenure_Status": tenure,
            "Electricity": electricity,
            "Type_of_Building_or_House": building,
            "Type_of_Roof": roof,
            "Type_of_Walls": walls,
            "Toilet_Facilities": toilet,
            "Main_Source_of_Water_Supply": water,

            # numerics
            "Total_Household_Income": income,
            "Total_Food_Expenditure": food_exp,
            "Food_Share": food_share,
            "Agricultural_Household_indicator": agri_hh,
            "Bread_and_Cereals_Expenditure": bread_cereals,
            "Total_Rice_Expenditure": rice_exp,
            "Meat_Expenditure": meat_exp,
            "Total_Fish_and__marine_products_Expenditure": fish_exp,
            "Fruit_Expenditure": fruit_exp,
            "Vegetables_Expenditure": veg_exp,
            "Restaurant_and_hotels_Expenditure": resto_exp,
            "Alcoholic_Beverages_Expenditure": alc_exp,
            "Tobacco_Expenditure": tob_exp,
            "Clothing,_Footwear_and_Other_Wear_Expenditure": cloth_exp,
            "Housing_and_water_Expenditure": house_water_exp,
            "Imputed_House_Rental_Value": imputed_rent,
            "Medical_Care_Expenditure": med_exp,
            "Transportation_Expenditure": trans_exp,
            "Communication_Expenditure": comm_exp,
            "Education_Expenditure": edu_exp,
            "Miscellaneous_Goods_and_Services_Expenditure": misc_exp,
            "Special_Occasions_Expenditure": special_exp,
            "Crop_Farming_and_Gardening_expenses": crop_exp,
            "Total_Income_from_Entrepreneurial_Acitivites": entrepreneurial_inc,
            "Household_Head_Age": head_age,
            "Total_Number_of_Family_members": hh_size,
            "Members_with_age_less_than_5_year_old": under5,
            "Members_with_age_5_-_17_years_old": age5_17,
            "Total_number_of_family_members_employed": employed_members,
            "House_Floor_Area": floor_area,
            "House_Age": house_age,
            "Number_of_bedrooms": bedrooms,
            "Number_of_Television": tv_count,
            "Number_of_CD/VCD/DVD": dvd_count,
            "Number_of_Component/Stereo_set": stereo_count,
            "Number_of_Refrigerator/Freezer": fridge_count,
            "Number_of_Washing_Machine": washer_count,
            "Number_of_Airconditioner": ac_count,
            "Number_of_Car,_Jeep,_Van": car_count,
            "Number_of_Landline/wireless_telephones": phone_landline,
            "Number_of_Cellular_phone": phone_cell,
            "Number_of_Personal_Computer": pc_count,
            "Number_of_Stove_with_Oven/Gas_Range": stove_count,
            "Number_of_Motorized_Banca": banca_count,
            "Number_of_Motorcycle/Tricycle": motorcycle_count
        }])

        clf  = st.session_state["fitted_clf"]
        if st.session_state["algo"].startswith("CatBoost"):
            for col in cat_cols:
                sample[col] = sample[col].astype(str)
        prob = clf.predict_proba(sample)[:,1][0]
        st.success(f"Predicted poverty probability: {prob:.2%}")
        st.progress(prob)

st.caption("Data: PSA FIES 2021 • Dashboard built with Streamlit & CatBoost")
