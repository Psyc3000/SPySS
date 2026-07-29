import io
import contextlib
import traceback

import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Student Stats Lab", page_icon="📊", layout="wide")
st.title("Student Stats Lab")
st.caption("A minimal SPSS-style interface powered by pandas and Pingouin.")

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".sav"):
        return pd.read_spss(uploaded_file)
    raise ValueError("Supported formats: CSV, XLSX/XLS, and SAV.")

with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls", "sav"])
    use_demo = st.checkbox("Use demonstration data", value=uploaded is None)

if uploaded is not None:
    try:
        df = load_data(uploaded)
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()
elif use_demo:
    df = pg.read_dataset("penguins").copy()
    source_name = "Pingouin penguins dataset"
else:
    st.info("Upload a dataset or enable the demonstration data.")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
all_cols = df.columns.tolist()

with st.sidebar:
    st.success(f"Loaded {source_name}")
    st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
    show_data = st.checkbox("Show data preview", value=True)

if show_data:
    st.subheader("Data preview")
    st.dataframe(df.head(100), use_container_width=True)

ANALYSES = {
    "1. Descriptive statistics": ("Counts, means, SDs, quartiles, skewness, and summaries.", 'result = pg.describe(df[["{x}", "{y}"]], percentiles=[0.25, 0.50, 0.75])', "two_numeric"),
    "2. Normality and Q–Q plot": ("Shapiro–Wilk normality test and a Q–Q plot.", 'result = pg.normality(df["{x}"], method="shapiro")\npg.qqplot(df["{x}"].dropna(), dist="norm")\nplt.title("Q–Q plot: {x}")', "one_numeric_plot"),
    "3. Correlation": ("Pearson, Spearman, Kendall, or robust correlation.", 'result = pg.corr(df["{x}"], df["{y}"], method="{method}")', "correlation"),
    "4. Partial correlation": ("Correlation while controlling for covariates.", 'result = pg.partial_corr(data=df, x="{x}", y="{y}", covar={covars}, method="{method}")', "partial"),
    "5. T-test": ("Paired or independent t-test.", 'result = pg.ttest(df["{x}"].dropna(), df["{y}"].dropna(), paired={paired}, alternative="{alternative}")', "ttest"),
    "6. One-way ANOVA": ("ANOVA with effect size and post hoc tests.", 'result = pg.anova(data=df, dv="{dv}", between="{between}", detailed=True)\nposthoc = pg.pairwise_{posthoc}(data=df, dv="{dv}", between="{between}")', "anova"),
    "7. Repeated-measures ANOVA": ("One-factor repeated-measures ANOVA in long format.", 'result = pg.rm_anova(data=df, dv="{dv}", within="{within}", subject="{subject}", detailed=True, correction=True)\nposthoc = pg.pairwise_tests(data=df, dv="{dv}", within="{within}", subject="{subject}", padjust="holm")', "rm"),
    "8. ANCOVA": ("ANCOVA with one grouping variable and covariates.", 'result = pg.ancova(data=df, dv="{dv}", between="{between}", covar={covars})', "ancova"),
    "9. Nonparametric tests": ("Mann–Whitney, Wilcoxon, Kruskal–Wallis, or Friedman.", '{nonparam_code}', "nonparam"),
    "10. Linear regression": ("Simple or multiple linear regression.", 'result = pg.linear_regression(df[{predictors}], df["{dv}"], add_intercept=True)', "regression"),
    "11. Reliability": ("Cronbach's alpha for a multi-item scale.", 'result = pg.cronbach_alpha(data=df[{items}], nan_policy="pairwise")', "reliability"),
    "12. Chi-square": ("Chi-square test of independence.", 'expected, observed, result = pg.chi2_independence(data=df, x="{x}", y="{y}", correction=True)', "chi2"),
}

st.sidebar.header("2. Analysis")
analysis_name = st.sidebar.selectbox("Choose an analysis", list(ANALYSES))
description, template, needs = ANALYSES[analysis_name]
st.header(analysis_name)
st.write(description)

def require(options, label):
    if not options:
        st.error(f"No suitable columns are available for {label}.")
        st.stop()

params = {}
if needs in {"two_numeric", "correlation", "ttest"}:
    require(numeric_cols, "this analysis")
    c1, c2 = st.columns(2)
    params["x"] = c1.selectbox("Variable X", numeric_cols, key=f"{needs}_x")
    params["y"] = c2.selectbox("Variable Y", numeric_cols, index=min(1, len(numeric_cols)-1), key=f"{needs}_y")
if needs == "one_numeric_plot":
    require(numeric_cols, "normality testing")
    params["x"] = st.selectbox("Numeric variable", numeric_cols)
if needs == "correlation":
    params["method"] = st.selectbox("Method", ["pearson", "spearman", "kendall", "bicor", "percbend", "shepherd", "skipped"])
if needs == "partial":
    require(numeric_cols, "partial correlation")
    c1, c2 = st.columns(2)
    params["x"] = c1.selectbox("Variable X", numeric_cols)
    params["y"] = c2.selectbox("Variable Y", numeric_cols, index=min(1, len(numeric_cols)-1))
    covs = st.multiselect("Covariate(s)", [c for c in numeric_cols if c not in {params['x'], params['y']}])
    params["covars"] = repr(covs if len(covs) > 1 else (covs[0] if covs else ""))
    params["method"] = st.selectbox("Method", ["pearson", "spearman"])
if needs == "ttest":
    params["paired"] = str(st.checkbox("Paired samples", value=False))
    params["alternative"] = st.selectbox("Alternative", ["two-sided", "greater", "less"])
if needs in {"anova", "ancova"}:
    require(numeric_cols, "dependent variable")
    c1, c2 = st.columns(2)
    params["dv"] = c1.selectbox("Dependent variable", numeric_cols)
    params["between"] = c2.selectbox("Grouping variable", [c for c in all_cols if c != params['dv']])
    if needs == "anova":
        params["posthoc"] = st.selectbox("Post hoc", ["tukey", "gameshowell"])
    else:
        covs = st.multiselect("Covariate(s)", [c for c in numeric_cols if c != params['dv']])
        params["covars"] = repr(covs if len(covs) > 1 else (covs[0] if covs else ""))
if needs == "rm":
    require(numeric_cols, "repeated-measures ANOVA")
    params["dv"] = st.selectbox("Dependent variable", numeric_cols)
    params["within"] = st.selectbox("Within-subject factor", [c for c in all_cols if c != params['dv']])
    params["subject"] = st.selectbox("Subject ID", [c for c in all_cols if c not in {params['dv'], params['within']}])
if needs == "nonparam":
    test = st.selectbox("Test", ["Mann–Whitney U", "Wilcoxon", "Kruskal–Wallis", "Friedman"])
    if test in {"Mann–Whitney U", "Wilcoxon"}:
        c1, c2 = st.columns(2)
        x = c1.selectbox("Variable X", numeric_cols)
        y = c2.selectbox("Variable Y", numeric_cols, index=min(1, len(numeric_cols)-1))
        fun = "mwu" if test == "Mann–Whitney U" else "wilcoxon"
        params["nonparam_code"] = f'result = pg.{fun}(df["{x}"].dropna(), df["{y}"].dropna())'
    elif test == "Kruskal–Wallis":
        dv = st.selectbox("Dependent variable", numeric_cols)
        between = st.selectbox("Grouping variable", [c for c in all_cols if c != dv])
        params["nonparam_code"] = f'result = pg.kruskal(data=df, dv="{dv}", between="{between}")'
    else:
        dv = st.selectbox("Dependent variable", numeric_cols)
        within = st.selectbox("Within-subject factor", [c for c in all_cols if c != dv])
        subject = st.selectbox("Subject ID", [c for c in all_cols if c not in {dv, within}])
        params["nonparam_code"] = f'result = pg.friedman(data=df, dv="{dv}", within="{within}", subject="{subject}")'
if needs == "regression":
    require(numeric_cols, "linear regression")
    params["dv"] = st.selectbox("Outcome", numeric_cols)
    params["predictors"] = repr(st.multiselect("Predictor(s)", [c for c in numeric_cols if c != params['dv']]))
if needs == "reliability":
    require(numeric_cols, "reliability")
    params["items"] = repr(st.multiselect("Scale items", numeric_cols))
if needs == "chi2":
    params["x"] = st.selectbox("Row variable", all_cols)
    params["y"] = st.selectbox("Column variable", [c for c in all_cols if c != params['x']])

try:
    default_code = template.format(**params)
except Exception:
    default_code = template

st.subheader("Editable Python code")
st.caption("The dataset is named `df`. Keep the main output in `result`.")
code = st.text_area("Code", value=default_code, height=170, label_visibility="collapsed")

if st.button("Run analysis", type="primary"):
    namespace = {"df": df.copy(), "pd": pd, "np": np, "pg": pg, "plt": plt, "sns": sns}
    stdout = io.StringIO()
    try:
        plt.close("all")
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": {"print": print, "len": len, "range": range}}, namespace)
        if stdout.getvalue():
            st.subheader("Console")
            st.code(stdout.getvalue())
        st.subheader("Output")
        displayed = False
        for name in ["result", "posthoc", "observed", "expected"]:
            if name in namespace:
                obj = namespace[name]
                st.markdown(f"**{name}**")
                if isinstance(obj, pd.DataFrame):
                    st.dataframe(obj, use_container_width=True)
                    st.download_button(f"Download {name}.csv", obj.to_csv(index=True).encode(), file_name=f"{name}.csv", key=f"dl_{name}")
                else:
                    st.write(obj)
                displayed = True
        for num in plt.get_fignums():
            st.pyplot(plt.figure(num), clear_figure=False)
            displayed = True
        if not displayed:
            st.info("Code ran, but no named table or plot was created.")
    except Exception:
        st.error("The analysis could not be completed.")
        st.code(traceback.format_exc())

with st.expander("Student notes"):
    st.markdown("""
- `df` is the uploaded dataset; column names are case-sensitive.
- Repeated-measures analyses require long-format data.
- Check assumptions and study design before interpreting p-values.
- The editable runner is for a trusted classroom workspace, not an unrestricted public site.
""")
