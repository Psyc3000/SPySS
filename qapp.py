from pathlib import Path
import traceback

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns

BASE = Path(__file__).parent

DATASETS = {
    "General teaching data": ("general_data.csv", "Descriptives, normality, correlations, t-tests, ANOVA, ANCOVA, and regression."),
    "Repeated-measures data": ("repeated_measures_data.csv", "Repeated-measures ANOVA and Friedman: score, time, participant."),
    "Reliability data": ("reliability_data.csv", "Cronbach alpha: select item1 through item8."),
    "Categorical data": ("categorical_data.csv", "Chi-square: treatment × outcome, adherence × outcome, or smoker × outcome."),
    "Nonparametric data": ("nonparametric_data.csv", "Mann–Whitney, Kruskal–Wallis, and Wilcoxon."),
    "Penguins (Pingouin)": (None, "A familiar dataset for descriptives, correlations, ANOVA, and regression."),
}

ANALYSES = [
    "1. Descriptive statistics",
    "2. Normality and Q-Q plot",
    "3. Correlation",
    "4. Partial correlation",
    "5. T-test",
    "6. One-way ANOVA",
    "7. Repeated-measures ANOVA",
    "8. ANCOVA",
    "9. Nonparametric test",
    "10. Linear regression",
    "11. Cronbach alpha",
    "12. Chi-square test",
]


def load_data(dataset_name, uploaded):
    if uploaded:
        path = Path(uploaded)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif suffix == ".sav":
            df = pd.read_spss(path)
        else:
            raise ValueError("Upload a CSV, Excel, or SPSS .sav file.")
        note = f"Uploaded file: {path.name}"
    else:
        filename, note = DATASETS[dataset_name]
        df = pg.read_dataset("penguins") if filename is None else pd.read_csv(BASE / filename)

    df.columns = [str(c).strip() for c in df.columns]
    return df, note


def defaults_for(dataset_name, all_cols, num_cols):
    def choose(name, pool, fallback=0):
        if name in pool:
            return name
        return pool[min(fallback, len(pool) - 1)] if pool else None

    x = choose("score", num_cols)
    y = choose("age", num_cols, 1)
    group = choose("group", all_cols)
    subject = choose("participant", all_cols)
    within = choose("time", all_cols)
    dv = choose("score", num_cols)
    covars = [c for c in ["age", "stress"] if c in num_cols]
    items = [c for c in num_cols if c.lower().startswith("item")]

    if dataset_name == "General teaching data":
        x = choose("pre_score", num_cols)
        y = choose("post_score", num_cols, 1)
        dv = choose("outcome", num_cols)
    elif dataset_name == "Reliability data":
        x = choose("item1", num_cols)
        y = choose("item2", num_cols, 1)
    elif dataset_name == "Categorical data":
        group = choose("treatment", all_cols)
        within = choose("outcome", all_cols, 1)
    elif dataset_name == "Nonparametric data":
        x = choose("before", num_cols)
        y = choose("after", num_cols, 1)
        dv = choose("skewed_score", num_cols)

    return x, y, group, subject, within, dv, covars, items


def refresh(dataset_name, uploaded, analysis):
    try:
        df, note = load_data(dataset_name, uploaded)
        all_cols = list(df.columns)
        num_cols = list(df.select_dtypes(include=np.number).columns)
        x, y, group, subject, within, dv, covars, items = defaults_for(dataset_name, all_cols, num_cols)
        return (
            note,
            df.head(50),
            gr.update(choices=num_cols, value=x),
            gr.update(choices=num_cols, value=y),
            gr.update(choices=all_cols, value=group),
            gr.update(choices=all_cols, value=subject),
            gr.update(choices=all_cols, value=within),
            gr.update(choices=num_cols, value=dv),
            gr.update(choices=num_cols, value=[]),
            gr.update(choices=num_cols, value=covars),
            gr.update(choices=num_cols, value=items),
        )
    except Exception:
        blank = gr.update(choices=[], value=None)
        return traceback.format_exc(), pd.DataFrame(), blank, blank, blank, blank, blank, blank, blank, blank, blank


def build_code(analysis, x, y, group, subject, within, dv, selected_vars, covars, items,
               method, t_design, test_value, anova_type, posthoc_type, nonparametric_test):
    selected_vars = selected_vars or []
    covars = covars or []
    items = items or []

    if analysis.startswith("1."):
        vars_ = selected_vars or [c for c in [x, y] if c]
        return "\n".join([
            f"x = df[{vars_!r}]",
            "result = pd.DataFrame({",
            '    "N": x.count(),',
            '    "Missing": x.isna().sum(),',
            '    "Mean": x.mean(),',
            '    "SD": x.std(),',
            '    "Median": x.median(),',
            '    "Minimum": x.min(),',
            '    "Maximum": x.max(),',
            '    "Skewness": x.skew(),',
            '    "Kurtosis": x.kurt()',
            "})",
        ])

    if analysis.startswith("2."):
        return "\n".join([
            f"result = pg.normality(df[{x!r}].dropna())",
            f"sns.histplot(df[{x!r}].dropna(), kde=True)",
            f'plt.title("Distribution of {x}")',
            "plt.show()",
            f"pg.qqplot(df[{x!r}].dropna())",
            f'plt.title("Q-Q plot of {x}")',
            "plt.show()",
        ])

    if analysis.startswith("3."):
        return "\n".join([
            f"result = pg.corr(x=df[{x!r}], y=df[{y!r}], method={method!r})",
            f"sns.regplot(data=df, x={x!r}, y={y!r})",
            "plt.show()",
        ])

    if analysis.startswith("4."):
        return f"result = pg.partial_corr(data=df, x={x!r}, y={y!r}, covar={covars!r})"

    if analysis.startswith("5."):
        if t_design == "One sample":
            return f"result = pg.ttest(df[{x!r}].dropna(), {float(test_value)})"
        if t_design == "Paired":
            return f"d = df[[{x!r}, {y!r}]].dropna()\nresult = pg.ttest(d[{x!r}], d[{y!r}], paired=True)"
        return "\n".join([
            f"levels = list(df[{group!r}].dropna().unique())",
            'if len(levels) != 2: raise ValueError("Independent t-test requires exactly two groups.")',
            f"a = df.loc[df[{group!r}] == levels[0], {dv!r}].dropna()",
            f"b = df.loc[df[{group!r}] == levels[1], {dv!r}].dropna()",
            "result = pg.ttest(a, b)",
        ])

    if analysis.startswith("6."):
        code = (f"result = pg.anova(data=df, dv={dv!r}, between={group!r}, detailed=True)"
                if anova_type == "Classical"
                else f"result = pg.welch_anova(data=df, dv={dv!r}, between={group!r})")
        if posthoc_type == "Tukey":
            code += f"\nposthoc = pg.pairwise_tukey(data=df, dv={dv!r}, between={group!r})"
        elif posthoc_type == "Games-Howell":
            code += f"\nposthoc = pg.pairwise_gameshowell(data=df, dv={dv!r}, between={group!r})"
        return code

    if analysis.startswith("7."):
        return "\n".join([
            f"d = df[[{subject!r}, {within!r}, {dv!r}]].dropna().copy()",
            f"counts = d.groupby({subject!r})[{within!r}].nunique()",
            "valid = counts[counts >= 2].index",
            f"d = d[d[{subject!r}].isin(valid)]",
            f'if d.empty or d[{within!r}].nunique() < 2: raise ValueError("No valid repeated observations.")',
            f"result = pg.rm_anova(data=d, dv={dv!r}, within={within!r}, subject={subject!r}, detailed=True)",
            f"posthoc = pg.pairwise_tests(data=d, dv={dv!r}, within={within!r}, subject={subject!r}, padjust='holm')",
        ])

    if analysis.startswith("8."):
        return f"result = pg.ancova(data=df, dv={dv!r}, between={group!r}, covar={covars!r})"

    if analysis.startswith("9."):
        if nonparametric_test == "Mann-Whitney":
            return "\n".join([
                f"levels = list(df[{group!r}].dropna().unique())",
                'if len(levels) < 2: raise ValueError("At least two groups are required.")',
                f"a = df.loc[df[{group!r}] == levels[0], {dv!r}].dropna()",
                f"b = df.loc[df[{group!r}] == levels[1], {dv!r}].dropna()",
                "result = pg.mwu(a, b)",
            ])
        if nonparametric_test == "Wilcoxon":
            return f"d = df[[{x!r}, {y!r}]].dropna()\nresult = pg.wilcoxon(d[{x!r}], d[{y!r}])"
        if nonparametric_test == "Kruskal-Wallis":
            return f"result = pg.kruskal(data=df, dv={dv!r}, between={group!r})"
        return f"d = df[[{subject!r}, {within!r}, {dv!r}]].dropna()\nresult = pg.friedman(data=d, dv={dv!r}, within={within!r}, subject={subject!r})"

    if analysis.startswith("10."):
        predictors = selected_vars or [x]
        return f"result = pg.linear_regression(X=df[{predictors!r}], y=df[{dv!r}], remove_na=True)"

    if analysis.startswith("11."):
        return "\n".join([
            f"alpha, ci = pg.cronbach_alpha(data=df[{items!r}])",
            'result = pd.DataFrame({"Cronbach alpha": [alpha], "CI lower": [ci[0]], "CI upper": [ci[1]]})',
        ])

    return f"expected, observed, result = pg.chi2_independence(data=df, x={group!r}, y={within!r})"


def run_code(dataset_name, uploaded, code):
    try:
        df, _ = load_data(dataset_name, uploaded)
        env = {"df": df.copy(), "pd": pd, "np": np, "pg": pg, "sns": sns, "plt": plt}
        plt.close("all")
        exec(code, env)
        outputs = []
        for name in ["result", "posthoc", "observed", "expected"]:
            value = env.get(name, pd.DataFrame())
            if isinstance(value, pd.Series):
                value = value.to_frame()
            elif not isinstance(value, pd.DataFrame):
                value = pd.DataFrame({"value": [value]})
            outputs.append(value)
        figures = [plt.figure(n) for n in plt.get_fignums()]
        return (*outputs, figures, "Analysis completed.")
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], traceback.format_exc()


with gr.Blocks(title="Student Statistics Lab — Gradio") as demo:
    gr.Markdown("# Student Statistics Lab — Gradio\nSame datasets, analyses, editable generated code, and outputs as the Streamlit version.")

    with gr.Row():
        with gr.Column(scale=1):
            dataset = gr.Dropdown(list(DATASETS), value="General teaching data", label="Built-in teaching dataset")
            upload = gr.File(label="Or upload CSV, Excel, or SAV", file_types=[".csv", ".xlsx", ".xls", ".sav"], type="filepath")
            dataset_note = gr.Markdown()
            analysis = gr.Dropdown(ANALYSES, value=ANALYSES[0], label="Analysis")
        with gr.Column(scale=2):
            preview = gr.Dataframe(label="Data preview", interactive=False, max_height=330)

    with gr.Accordion("Variables and options", open=True):
        with gr.Row():
            x = gr.Dropdown(label="Variable X / Variable 1")
            y = gr.Dropdown(label="Variable Y / Variable 2")
            dv = gr.Dropdown(label="Dependent/outcome variable")
        with gr.Row():
            group = gr.Dropdown(label="Grouping/factor/row variable")
            within = gr.Dropdown(label="Within factor/column variable")
            subject = gr.Dropdown(label="Subject ID")
        with gr.Row():
            selected_vars = gr.Dropdown(multiselect=True, label="Variables / predictors")
            covars = gr.Dropdown(multiselect=True, label="Covariates")
            items = gr.Dropdown(multiselect=True, label="Scale items")
        with gr.Row():
            method = gr.Dropdown(["pearson", "spearman", "kendall", "bicor"], value="pearson", label="Correlation method")
            t_design = gr.Dropdown(["One sample", "Paired", "Independent"], value="Independent", label="T-test design")
            test_value = gr.Number(value=0, label="One-sample test value")
        with gr.Row():
            anova_type = gr.Dropdown(["Classical", "Welch"], value="Classical", label="ANOVA type")
            posthoc_type = gr.Dropdown(["None", "Tukey", "Games-Howell"], value="None", label="Post hoc")
            nonparametric_test = gr.Dropdown(["Mann-Whitney", "Wilcoxon", "Kruskal-Wallis", "Friedman"], value="Mann-Whitney", label="Nonparametric test")

    generate = gr.Button("Generate code")
    code = gr.Code(label="Editable Python code", language="python", lines=18)
    run = gr.Button("Run analysis", variant="primary")
    status = gr.Textbox(label="Status / error", lines=8)

    with gr.Tabs():
        with gr.Tab("Result"):
            result = gr.Dataframe(interactive=False)
        with gr.Tab("Post hoc"):
            posthoc = gr.Dataframe(interactive=False)
        with gr.Tab("Observed"):
            observed = gr.Dataframe(interactive=False)
        with gr.Tab("Expected"):
            expected = gr.Dataframe(interactive=False)
        with gr.Tab("Plots"):
            plots = gr.Gallery(label="Plots", columns=2, object_fit="contain")

    refresh_inputs = [dataset, upload, analysis]
    refresh_outputs = [dataset_note, preview, x, y, group, subject, within, dv, selected_vars, covars, items]
    demo.load(refresh, refresh_inputs, refresh_outputs)
    dataset.change(refresh, refresh_inputs, refresh_outputs)
    upload.change(refresh, refresh_inputs, refresh_outputs)
    analysis.change(refresh, refresh_inputs, refresh_outputs)

    code_inputs = [analysis, x, y, group, subject, within, dv, selected_vars, covars, items,
                   method, t_design, test_value, anova_type, posthoc_type, nonparametric_test]
    generate.click(build_code, code_inputs, code)
    run.click(run_code, [dataset, upload, code], [result, posthoc, observed, expected, plots, status])

if __name__ == "__main__":
    demo.launch()
