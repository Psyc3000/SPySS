import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Student Statistics Lab", layout="wide")
st.title("Student Statistics Lab")
st.caption("pandas for descriptives • seaborn/matplotlib for plots • Pingouin for inference")

# Load data
uploaded = st.sidebar.file_uploader("Upload CSV, Excel, or SAV", type=["csv", "xlsx", "xls", "sav"])
if uploaded:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_spss(uploaded)
else:
    df = pg.read_dataset("penguins")

df.columns = [str(c).strip() for c in df.columns]
all_cols = list(df.columns)
num_cols = list(df.select_dtypes(include=np.number).columns)
st.sidebar.write(f"{len(df)} rows × {len(df.columns)} columns")

if st.checkbox("Show data preview", value=True):
    st.dataframe(df.head(50), use_container_width=True)

analysis = st.sidebar.selectbox("Analysis", [
    "1. Descriptive statistics", "2. Normality and Q-Q plot", "3. Correlation",
    "4. Partial correlation", "5. T-test", "6. One-way ANOVA",
    "7. Repeated-measures ANOVA", "8. ANCOVA", "9. Nonparametric test",
    "10. Linear regression", "11. Cronbach alpha", "12. Chi-square test"
])
st.header(analysis)
code = ""

if analysis.startswith("1."):
    vars_ = st.multiselect("Numeric variables", num_cols, default=num_cols[:4])
    code = f'''x = df[{vars_!r}]
result = pd.DataFrame({{
    "N": x.count(), "Missing": x.isna().sum(), "Mean": x.mean(),
    "SD": x.std(), "Median": x.median(), "Minimum": x.min(),
    "Maximum": x.max(), "Skewness": x.skew(), "Kurtosis": x.kurt()
}})'''

elif analysis.startswith("2."):
    x = st.selectbox("Variable", num_cols)
    code = f'''result = pg.normality(df[{x!r}].dropna())
sns.histplot(df[{x!r}].dropna(), kde=True)
plt.title("Distribution of {x}")
plt.show()
pg.qqplot(df[{x!r}].dropna())
plt.title("Q-Q plot of {x}")
plt.show()'''

elif analysis.startswith("3."):
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X", num_cols)
    y = c2.selectbox("Y", num_cols, index=min(1, len(num_cols) - 1))
    method = c3.selectbox("Method", ["pearson", "spearman", "kendall", "bicor"])
    code = f'''result = pg.corr(x=df[{x!r}], y=df[{y!r}], method={method!r})
sns.regplot(data=df, x={x!r}, y={y!r})
plt.show()'''

elif analysis.startswith("4."):
    x = st.selectbox("X", num_cols)
    y = st.selectbox("Y", num_cols, index=min(1, len(num_cols) - 1))
    covars = st.multiselect("Covariates", [c for c in num_cols if c not in [x, y]])
    code = f'''result = pg.partial_corr(data=df, x={x!r}, y={y!r}, covar={covars!r})'''

elif analysis.startswith("5."):
    design = st.radio("Design", ["One sample", "Paired", "Independent"], horizontal=True)
    if design == "One sample":
        x = st.selectbox("Variable", num_cols)
        mu = st.number_input("Test value", value=0.0)
        code = f'''result = pg.ttest(df[{x!r}].dropna(), {mu})'''
    elif design == "Paired":
        x = st.selectbox("Variable 1", num_cols)
        y = st.selectbox("Variable 2", num_cols, index=min(1, len(num_cols) - 1))
        code = f'''d = df[[{x!r}, {y!r}]].dropna()
result = pg.ttest(d[{x!r}], d[{y!r}], paired=True)'''
    else:
        outcome = st.selectbox("Outcome", num_cols)
        group = st.selectbox("Grouping variable", [c for c in all_cols if c != outcome])
        levels = list(df[group].dropna().unique())
        chosen = st.multiselect("Choose two groups", levels, default=levels[:2])
        code = "# Choose exactly two groups."
        if len(chosen) == 2:
            g1, g2 = chosen
            code = f'''x = df.loc[df[{group!r}] == {g1!r}, {outcome!r}].dropna()
y = df.loc[df[{group!r}] == {g2!r}, {outcome!r}].dropna()
result = pg.ttest(x, y)'''

elif analysis.startswith("6."):
    dv = st.selectbox("Dependent variable", num_cols)
    between = st.selectbox("Factor", [c for c in all_cols if c != dv])
    kind = st.radio("Type", ["Classical", "Welch"], horizontal=True)
    posthoc = st.selectbox("Post hoc", ["None", "Tukey", "Games-Howell"])
    code = (f'''result = pg.anova(data=df, dv={dv!r}, between={between!r}, detailed=True)'''
            if kind == "Classical" else
            f'''result = pg.welch_anova(data=df, dv={dv!r}, between={between!r})''')
    if posthoc == "Tukey":
        code += f'''\nposthoc = pg.pairwise_tukey(data=df, dv={dv!r}, between={between!r})'''
    elif posthoc == "Games-Howell":
        code += f'''\nposthoc = pg.pairwise_gameshowell(data=df, dv={dv!r}, between={between!r})'''

elif analysis.startswith("7."):
    dv = st.selectbox("Dependent variable", num_cols)
    within = st.selectbox("Within-subject factor", [c for c in all_cols if c != dv])
    subject = st.selectbox("Subject ID", [c for c in all_cols if c not in [dv, within]])
    code = f'''result = pg.rm_anova(data=df, dv={dv!r}, within={within!r}, subject={subject!r}, detailed=True)
posthoc = pg.pairwise_tests(data=df, dv={dv!r}, within={within!r}, subject={subject!r}, paired=True, padjust="holm")'''

elif analysis.startswith("8."):
    dv = st.selectbox("Dependent variable", num_cols)
    between = st.selectbox("Factor", [c for c in all_cols if c != dv])
    covars = st.multiselect("Covariates", [c for c in num_cols if c != dv])
    code = f'''result = pg.ancova(data=df, dv={dv!r}, between={between!r}, covar={covars!r})'''

elif analysis.startswith("9."):
    test = st.selectbox("Test", ["Mann-Whitney", "Wilcoxon", "Kruskal-Wallis", "Friedman"])
    if test == "Mann-Whitney":
        outcome = st.selectbox("Outcome", num_cols)
        group = st.selectbox("Grouping variable", [c for c in all_cols if c != outcome])
        levels = list(df[group].dropna().unique())
        chosen = st.multiselect("Choose two groups", levels, default=levels[:2])
        code = "# Choose exactly two groups."
        if len(chosen) == 2:
            g1, g2 = chosen
            code = f'''x = df.loc[df[{group!r}] == {g1!r}, {outcome!r}].dropna()
y = df.loc[df[{group!r}] == {g2!r}, {outcome!r}].dropna()
result = pg.mwu(x, y)'''
    elif test == "Wilcoxon":
        x = st.selectbox("Variable 1", num_cols)
        y = st.selectbox("Variable 2", num_cols, index=min(1, len(num_cols) - 1))
        code = f'''d = df[[{x!r}, {y!r}]].dropna()
result = pg.wilcoxon(d[{x!r}], d[{y!r}])'''
    elif test == "Kruskal-Wallis":
        dv = st.selectbox("Dependent variable", num_cols)
        between = st.selectbox("Factor", [c for c in all_cols if c != dv])
        code = f'''result = pg.kruskal(data=df, dv={dv!r}, between={between!r})'''
    else:
        dv = st.selectbox("Dependent variable", num_cols)
        within = st.selectbox("Within factor", [c for c in all_cols if c != dv])
        subject = st.selectbox("Subject ID", [c for c in all_cols if c not in [dv, within]])
        code = f'''result = pg.friedman(data=df, dv={dv!r}, within={within!r}, subject={subject!r})'''

elif analysis.startswith("10."):
    y = st.selectbox("Outcome", num_cols)
    xs = st.multiselect("Predictors", [c for c in num_cols if c != y])
    code = f'''result = pg.linear_regression(X=df[{xs!r}], y=df[{y!r}], remove_na=True)'''

elif analysis.startswith("11."):
    items = st.multiselect("Scale items", num_cols, default=num_cols[:3])
    code = f'''alpha, ci = pg.cronbach_alpha(data=df[{items!r}])
result = pd.DataFrame({{"Cronbach alpha": [alpha], "CI lower": [ci[0]], "CI upper": [ci[1]]}})'''

elif analysis.startswith("12."):
    x = st.selectbox("Row variable", all_cols)
    y = st.selectbox("Column variable", [c for c in all_cols if c != x])
    code = f'''expected, observed, result = pg.chi2_independence(data=df, x={x!r}, y={y!r})'''

st.subheader("Editable code")
code = st.text_area("Edit before running", code, height=210, label_visibility="collapsed")

if st.button("Run analysis", type="primary"):
    env = {"df": df.copy(), "pd": pd, "np": np, "pg": pg, "sns": sns, "plt": plt}
    try:
        plt.close("all")
        exec(code, env)
        for name in ["result", "posthoc", "observed", "expected"]:
            if name in env:
                st.subheader(name.capitalize())
                value = env[name]
                st.dataframe(value, use_container_width=True) if isinstance(value, (pd.DataFrame, pd.Series)) else st.write(value)
        for number in plt.get_fignums():
            st.pyplot(plt.figure(number))
    except Exception:
        st.error("Analysis failed")
        st.code(traceback.format_exc())
