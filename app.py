from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui


def sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 80
    group = rng.choice(["Control", "Treatment"], n)
    gender = rng.choice(["Female", "Male"], n)
    age = rng.integers(18, 55, n)
    study_hours = np.clip(rng.normal(6.2, 2.0, n), 1, 12)
    sleep_hours = np.clip(rng.normal(7.0, 1.0, n), 4, 10)
    stress = np.clip(rng.normal(5.4, 1.8, n), 1, 10)
    exam_score = 48 + 3.3 * study_hours + 1.4 * sleep_hours - 1.8 * stress + np.where(group == "Treatment", 4.5, 0) + rng.normal(0, 6, n)
    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "group": group,
        "gender": gender,
        "age": age,
        "study_hours": study_hours.round(1),
        "sleep_hours": sleep_hours.round(1),
        "stress": stress.round(1),
        "exam_score": exam_score.round(1),
    })


APP_CSS = """
:root { --spss-blue: #245a9a; }
.navbar { background: var(--spss-blue) !important; }
.navbar-brand, .navbar .nav-link { color: white !important; }
.card-header { font-weight: 650; }
.sidebar { background: #f7f9fc; }
.stat-note { color: #5b6470; font-size: .9rem; }
pre { white-space: pre-wrap; }
"""

app_ui = ui.page_navbar(
    ui.nav_panel("Data",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Data source"),
                ui.input_file("csv_file", "Upload CSV", accept=[".csv"]),
                ui.input_action_button("load_sample", "Reload sample data"),
                ui.hr(), ui.output_text("data_status"),
                ui.download_button("download_data", "Download current data"),
                width=300,
            ),
            ui.card(ui.card_header("Data Editor"), ui.p("Double-click a cell to edit it. Analyses use edited values.", class_="stat-note"), ui.output_data_frame("data_table"), full_screen=True),
            ui.card(ui.card_header("Variable View"), ui.output_data_frame("variable_view")),
        )
    ),
    ui.nav_panel("Descriptives",
        ui.layout_sidebar(
            ui.sidebar(ui.input_selectize("desc_vars", "Variables", choices=[], multiple=True), ui.input_checkbox("include_ci", "Include 95% confidence interval", True), width=300),
            ui.card(ui.card_header("Descriptive Statistics"), ui.output_data_frame("descriptive_table"), full_screen=True),
            ui.card(ui.card_header("Distribution"), ui.input_select("hist_var", "Variable", choices=[]), ui.output_plot("histogram")),
        )
    ),
    ui.nav_panel("Correlation",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("corr_x", "X variable", choices=[]),
                ui.input_select("corr_y", "Y variable", choices=[]),
                ui.input_radio_buttons("corr_method", "Method", {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall"}, selected="pearson"),
                width=300,
            ),
            ui.card(ui.card_header("Correlation Result"), ui.output_text_verbatim("correlation_result")),
            ui.card(ui.card_header("Scatterplot"), ui.output_plot("correlation_plot")),
        )
    ),
    ui.nav_panel("Regression",
        ui.layout_sidebar(
            ui.sidebar(ui.input_select("reg_y", "Dependent variable", choices=[]), ui.input_selectize("reg_x", "Predictors", choices=[], multiple=True), ui.input_checkbox("reg_intercept", "Include intercept", True), width=300),
            ui.card(ui.card_header("Model Summary"), ui.output_text_verbatim("regression_summary"), full_screen=True),
            ui.card(ui.card_header("Coefficients"), ui.output_data_frame("coefficient_table")),
            ui.card(ui.card_header("Residual Diagnostics"), ui.output_plot("residual_plot")),
        )
    ),
    ui.nav_panel("t Tests",
        ui.layout_sidebar(
            ui.sidebar(ui.input_select("ttest_type", "Test", {"independent": "Independent-samples t test", "paired": "Paired-samples t test", "one_sample": "One-sample t test"}), ui.output_ui("ttest_controls"), width=320),
            ui.card(ui.card_header("t Test Result"), ui.output_text_verbatim("ttest_result")),
        )
    ),
    ui.nav_panel("ANOVA",
        ui.layout_sidebar(
            ui.sidebar(ui.input_select("anova_y", "Dependent variable", choices=[]), ui.input_select("anova_group", "Factor", choices=[]), width=300),
            ui.card(ui.card_header("One-Way ANOVA"), ui.output_data_frame("anova_table")),
            ui.card(ui.card_header("Group Descriptives"), ui.output_data_frame("anova_descriptives")),
            ui.card(ui.card_header("Group Plot"), ui.output_plot("anova_plot")),
        )
    ),
    ui.nav_panel("Charts",
        ui.layout_sidebar(
            ui.sidebar(ui.input_select("chart_type", "Chart", {"scatter": "Scatterplot", "hist": "Histogram", "box": "Boxplot", "bar": "Bar chart"}), ui.output_ui("chart_controls"), width=300),
            ui.card(ui.card_header("Chart"), ui.output_plot("custom_chart"), full_screen=True),
        )
    ),
    ui.nav_panel("Export",
        ui.card(ui.card_header("Export Results"), ui.p("Download the edited dataset or a text report."), ui.layout_columns(ui.download_button("download_data_2", "Download data as CSV"), ui.download_button("download_report", "Download analysis report"), col_widths=[6, 6]))
    ),
    title="SPSS-like Statistics", header=ui.tags.style(APP_CSS), fillable=True,
)


def server(input: Inputs, output: Outputs, session: Session):
    dataset = reactive.Value(sample_data())

    @reactive.effect
    @reactive.event(input.csv_file)
    def _load_csv():
        info = input.csv_file(); req(info)
        try:
            df = pd.read_csv(info[0]["datapath"])
            if df.empty: raise ValueError("The CSV contains no rows.")
            dataset.set(df)
            ui.notification_show("CSV loaded.", type="message")
        except Exception as exc:
            ui.notification_show(f"Could not read CSV: {exc}", type="error")

    @reactive.effect
    @reactive.event(input.load_sample)
    def _reload_sample():
        dataset.set(sample_data())

    @render.data_frame
    def data_table():
        return render.DataGrid(dataset(), editable=True, filters=True, selection_mode="rows", height="520px")

    @data_table.set_patch_fn
    def _coerce_edit(*, patch):
        col = dataset().columns[patch["column_index"]]
        if pd.api.types.is_numeric_dtype(dataset()[col]):
            try: return pd.to_numeric(patch["value"])
            except Exception: return np.nan
        return patch["value"]

    @reactive.calc
    def current_data():
        try: return data_table.data_patched().copy()
        except Exception: return dataset().copy()

    @reactive.calc
    def numeric_columns():
        return current_data().select_dtypes(include=np.number).columns.tolist()

    @reactive.calc
    def categorical_columns():
        return [c for c in current_data().columns if c not in numeric_columns()]

    @reactive.effect
    def _update_choices():
        nums, cats = numeric_columns(), categorical_columns()
        first = lambda x: x[0] if x else None
        ui.update_selectize("desc_vars", choices=nums, selected=nums[:3])
        ui.update_select("hist_var", choices=nums, selected=first(nums))
        ui.update_select("corr_x", choices=nums, selected=first(nums))
        ui.update_select("corr_y", choices=nums, selected=nums[1] if len(nums) > 1 else first(nums))
        ui.update_select("reg_y", choices=nums, selected=nums[-1] if nums else None)
        ui.update_selectize("reg_x", choices=nums, selected=nums[:2])
        ui.update_select("anova_y", choices=nums, selected=nums[-1] if nums else None)
        ui.update_select("anova_group", choices=cats, selected=first(cats))

    @render.text
    def data_status():
        df = current_data(); return f"{df.shape[0]} cases × {df.shape[1]} variables"

    @render.data_frame
    def variable_view():
        df = current_data()
        rows = [{"Variable": c, "Type": str(df[c].dtype), "Numeric": pd.api.types.is_numeric_dtype(df[c]), "Missing": int(df[c].isna().sum()), "Unique": int(df[c].nunique(dropna=True))} for c in df.columns]
        return render.DataGrid(pd.DataFrame(rows), filters=True)

    @render.data_frame
    def descriptive_table():
        selected = list(input.desc_vars() or []); req(selected)
        df = current_data()[selected].apply(pd.to_numeric, errors="coerce")
        rows = []
        for col in selected:
            s = df[col].dropna(); n = len(s); se = s.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            row = {"Variable": col, "N": n, "Missing": int(df[col].isna().sum()), "Mean": s.mean(), "SD": s.std(ddof=1), "Median": s.median(), "Minimum": s.min(), "Maximum": s.max(), "Skewness": s.skew(), "Kurtosis": s.kurt()}
            if input.include_ci() and n > 1:
                crit = stats.t.ppf(.975, n - 1); row["95% CI Lower"] = s.mean() - crit * se; row["95% CI Upper"] = s.mean() + crit * se
            rows.append(row)
        return render.DataGrid(pd.DataFrame(rows).round(4), filters=True)

    @render.plot
    def histogram():
        var = input.hist_var(); req(var)
        s = pd.to_numeric(current_data()[var], errors="coerce").dropna()
        fig, ax = plt.subplots(); ax.hist(s, bins="auto", edgecolor="black"); ax.set(xlabel=var, ylabel="Frequency", title=f"Distribution of {var}"); fig.tight_layout(); return fig

    @render.text
    def correlation_result():
        x, y = input.corr_x(), input.corr_y(); req(x, y)
        d = current_data()[[x, y]].apply(pd.to_numeric, errors="coerce").dropna(); req(len(d) >= 3)
        method = input.corr_method()
        if method == "pearson": result, symbol = stats.pearsonr(d[x], d[y]), "r"
        elif method == "spearman": result, symbol = stats.spearmanr(d[x], d[y]), "ρ"
        else: result, symbol = stats.kendalltau(d[x], d[y]), "τ"
        return f"Method: {method.title()}\nN = {len(d)}\n{symbol} = {result.statistic:.4f}\np = {result.pvalue:.6g}"

    @render.plot
    def correlation_plot():
        x, y = input.corr_x(), input.corr_y(); req(x, y)
        d = current_data()[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        fig, ax = plt.subplots(); ax.scatter(d[x], d[y], alpha=.75)
        if len(d) >= 2:
            slope, intercept = np.polyfit(d[x], d[y], 1); xx = np.linspace(d[x].min(), d[x].max(), 100); ax.plot(xx, intercept + slope * xx)
        ax.set(xlabel=x, ylabel=y, title=f"{y} by {x}"); fig.tight_layout(); return fig

    @reactive.calc
    def fitted_model():
        y, xs = input.reg_y(), list(input.reg_x() or []); req(y, xs)
        xs = [x for x in xs if x != y]; req(xs)
        d = current_data()[[y] + xs].apply(pd.to_numeric, errors="coerce").dropna(); req(len(d) > len(xs) + 1)
        X = sm.add_constant(d[xs]) if input.reg_intercept() else d[xs]
        return sm.OLS(d[y], X).fit()

    @render.text
    def regression_summary(): return fitted_model().summary().as_text()

    @render.data_frame
    def coefficient_table():
        m = fitted_model(); ci = m.conf_int()
        out = pd.DataFrame({"Term": m.params.index, "B": m.params.values, "SE": m.bse.values, "t": m.tvalues.values, "p": m.pvalues.values, "CI Lower": ci[0].values, "CI Upper": ci[1].values})
        return render.DataGrid(out.round(5))

    @render.plot
    def residual_plot():
        m = fitted_model(); fig, ax = plt.subplots(); ax.scatter(m.fittedvalues, m.resid, alpha=.75); ax.axhline(0, linestyle="--"); ax.set(xlabel="Predicted values", ylabel="Residuals", title="Residuals vs Predicted"); fig.tight_layout(); return fig

    @render.ui
    def ttest_controls():
        nums, cats, test = numeric_columns(), categorical_columns(), input.ttest_type()
        if test == "independent": return ui.TagList(ui.input_select("ttest_y", "Outcome", choices=nums), ui.input_select("ttest_group", "Two-level grouping variable", choices=cats), ui.input_checkbox("ttest_equal_var", "Assume equal variances", False))
        if test == "paired": return ui.TagList(ui.input_select("ttest_a", "First variable", choices=nums), ui.input_select("ttest_b", "Second variable", choices=nums))
        return ui.TagList(ui.input_select("ttest_one_var", "Variable", choices=nums), ui.input_numeric("ttest_mu", "Test value", value=0))

    @render.text
    def ttest_result():
        test, df = input.ttest_type(), current_data()
        if test == "independent":
            y, g = input.ttest_y(), input.ttest_group(); req(y, g); d = df[[y, g]].dropna(); levels = d[g].unique(); req(len(levels) == 2)
            a = pd.to_numeric(d.loc[d[g] == levels[0], y], errors="coerce").dropna(); b = pd.to_numeric(d.loc[d[g] == levels[1], y], errors="coerce").dropna(); r = stats.ttest_ind(a, b, equal_var=input.ttest_equal_var())
            return f"{levels[0]}: n={len(a)}, M={a.mean():.3f}, SD={a.std(ddof=1):.3f}\n{levels[1]}: n={len(b)}, M={b.mean():.3f}, SD={b.std(ddof=1):.3f}\n\nt = {r.statistic:.4f}\np = {r.pvalue:.6g}"
        if test == "paired":
            a, b = input.ttest_a(), input.ttest_b(); req(a, b); d = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna(); r = stats.ttest_rel(d[a], d[b]); return f"N = {len(d)}\nt = {r.statistic:.4f}\np = {r.pvalue:.6g}"
        var = input.ttest_one_var(); req(var); s = pd.to_numeric(df[var], errors="coerce").dropna(); r = stats.ttest_1samp(s, popmean=input.ttest_mu()); return f"N = {len(s)}\nMean = {s.mean():.4f}\nt = {r.statistic:.4f}\np = {r.pvalue:.6g}"

    @reactive.calc
    def anova_model():
        y, g = input.anova_y(), input.anova_group(); req(y, g)
        d = current_data()[[y, g]].dropna().copy(); d[y] = pd.to_numeric(d[y], errors="coerce"); d = d.dropna(); req(d[g].nunique() >= 2)
        model = smf.ols(f'Q("{y}") ~ C(Q("{g}"))', data=d).fit(); return model, d, y, g

    @render.data_frame
    def anova_table():
        m, _, _, _ = anova_model(); t = sm.stats.anova_lm(m, typ=2).reset_index(); t.columns = ["Source", "Sum of Squares", "df", "F", "p"]; return render.DataGrid(t.round(5))

    @render.data_frame
    def anova_descriptives():
        _, d, y, g = anova_model(); out = d.groupby(g, dropna=False)[y].agg(N="count", Mean="mean", SD="std", Minimum="min", Maximum="max").reset_index(); return render.DataGrid(out.round(4))

    @render.plot
    def anova_plot():
        _, d, y, g = anova_model(); levels = list(d[g].astype(str).unique()); vals = [d.loc[d[g].astype(str) == lev, y] for lev in levels]
        fig, ax = plt.subplots(); ax.boxplot(vals, tick_labels=levels); ax.set(xlabel=g, ylabel=y, title=f"{y} by {g}"); fig.tight_layout(); return fig

    @render.ui
    def chart_controls():
        nums, cats, chart = numeric_columns(), categorical_columns(), input.chart_type()
        if chart == "scatter": return ui.TagList(ui.input_select("chart_x", "X", choices=nums), ui.input_select("chart_y", "Y", choices=nums))
        if chart == "hist": return ui.input_select("chart_hist", "Variable", choices=nums)
        if chart == "box": return ui.TagList(ui.input_select("chart_box_y", "Numeric variable", choices=nums), ui.input_select("chart_box_g", "Group", choices=cats))
        return ui.input_select("chart_bar", "Category", choices=cats)

    @render.plot
    def custom_chart():
        chart, df = input.chart_type(), current_data(); fig, ax = plt.subplots()
        if chart == "scatter":
            x, y = input.chart_x(), input.chart_y(); req(x, y); d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna(); ax.scatter(d[x], d[y], alpha=.75); ax.set(xlabel=x, ylabel=y, title=f"{y} by {x}")
        elif chart == "hist":
            v = input.chart_hist(); req(v); s = pd.to_numeric(df[v], errors="coerce").dropna(); ax.hist(s, bins="auto", edgecolor="black"); ax.set(xlabel=v, ylabel="Frequency", title=f"Histogram of {v}")
        elif chart == "box":
            y, g = input.chart_box_y(), input.chart_box_g(); req(y, g); d = df[[y, g]].dropna(); levels = list(d[g].astype(str).unique()); vals = [pd.to_numeric(d.loc[d[g].astype(str) == lev, y], errors="coerce").dropna() for lev in levels]; ax.boxplot(vals, tick_labels=levels); ax.set(xlabel=g, ylabel=y, title=f"{y} by {g}")
        else:
            v = input.chart_bar(); req(v); counts = df[v].astype(str).value_counts(); ax.bar(counts.index, counts.values); ax.tick_params(axis="x", rotation=35); ax.set(xlabel=v, ylabel="Count", title=f"Counts of {v}")
        fig.tight_layout(); return fig

    @render.download(filename="statistics_data.csv")
    def download_data(): yield current_data().to_csv(index=False)

    @render.download(filename="statistics_data.csv")
    def download_data_2(): yield current_data().to_csv(index=False)

    @render.download(filename="analysis_report.txt")
    def download_report():
        parts = ["SPSS-LIKE STATISTICS REPORT", "=" * 40, f"Cases: {current_data().shape[0]}", f"Variables: {current_data().shape[1]}", ""]
        try: parts += ["CORRELATION", correlation_result(), ""]
        except Exception: pass
        try: parts += ["REGRESSION", fitted_model().summary().as_text(), ""]
        except Exception: pass
        try: parts += ["T TEST", ttest_result(), ""]
        except Exception: pass
        yield "\n".join(parts)


app = App(app_ui, server)
