import contextlib
import io
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import streamlit as st

st.set_page_config(page_title='Student Statistics Lab', page_icon='📊', layout='wide')
st.title('Student Statistics Lab')
st.caption('A compact SPSS-style interface using pandas, Pingouin, seaborn, and matplotlib.')

@st.cache_data(show_spinner=False)
def read_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(uploaded)
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded)
    if name.endswith('.sav'):
        return pd.read_spss(uploaded)
    raise ValueError('Supported formats are CSV, Excel, and SPSS SAV.')

with st.sidebar:
    st.header('1. Data')
    uploaded = st.file_uploader('Upload data', type=['csv', 'xlsx', 'xls', 'sav'])
    use_demo = st.checkbox('Use demonstration data', value=uploaded is None)

if uploaded is not None:
    try:
        df = read_file(uploaded)
        data_name = uploaded.name
    except Exception as exc:
        st.error(f'File could not be read: {exc}')
        st.stop()
elif use_demo:
    df = pg.read_dataset('penguins').copy()
    data_name = 'Pingouin penguins dataset'
else:
    st.info('Upload a dataset or select the demonstration dataset.')
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

with st.sidebar:
    st.success(f'{data_name}\n\n{len(df):,} rows × {len(df.columns)} columns')
    preview = st.checkbox('Show data preview', value=True)

if preview:
    st.subheader('Data preview')
    st.dataframe(df.head(100), use_container_width=True)

def require(columns, message):
    if not columns:
        st.error(message)
        st.stop()

def q(name):
    return repr(str(name))

def list_code(values):
    return repr([str(v) for v in values])

analysis = st.sidebar.selectbox('2. Analysis', [
    '1. Descriptive statistics',
    '2. Normality and Q–Q plot',
    '3. Correlation',
    '4. Partial correlation',
    '5. T-test',
    '6. One-way ANOVA',
    '7. Repeated-measures ANOVA',
    '8. ANCOVA',
    '9. Nonparametric tests',
    '10. Linear regression',
    '11. Reliability',
    '12. Chi-square test',
])

st.header(analysis)
default_code = ''

if analysis == '1. Descriptive statistics':
    st.write('SPSS-style descriptive statistics from pandas.')
    require(numeric_cols, 'No numeric variables were found.')
    variables = st.multiselect('Variables', numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
    default_code = f'''variables = {list_code(variables)}
x = df[variables]
result = pd.DataFrame({{
    "N": x.count(),
    "Missing": x.isna().sum(),
    "Mean": x.mean(),
    "SD": x.std(ddof=1),
    "Variance": x.var(ddof=1),
    "Median": x.median(),
    "Minimum": x.min(),
    "Maximum": x.max(),
    "Skewness": x.skew(),
    "Kurtosis": x.kurt()
}})
'''

elif analysis == '2. Normality and Q–Q plot':
    st.write('Shapiro–Wilk normality test plus a Q–Q plot.')
    require(numeric_cols, 'No numeric variables were found.')
    x = st.selectbox('Variable', numeric_cols)
    default_code = f'''result = pg.normality(df[{q(x)}].dropna(), method="shapiro")
pg.qqplot(df[{q(x)}].dropna(), dist="norm")
plt.title("Q–Q plot: {x}")
'''

elif analysis == '3. Correlation':
    st.write('Pearson, rank, or robust bivariate correlation.')
    require(numeric_cols, 'At least two numeric variables are required.')
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox('Variable X', numeric_cols)
    y = c2.selectbox('Variable Y', numeric_cols, index=min(1, len(numeric_cols)-1))
    method = c3.selectbox('Method', ['pearson', 'spearman', 'kendall', 'bicor', 'percbend', 'shepherd', 'skipped'])
    default_code = f'''result = pg.corr(x=df[{q(x)}], y=df[{q(y)}], method={q(method)})'''

elif analysis == '4. Partial correlation':
    st.write('Correlation between two variables while controlling for covariates.')
    require(numeric_cols, 'At least three numeric variables are recommended.')
    c1, c2 = st.columns(2)
    x = c1.selectbox('Variable X', numeric_cols)
    y = c2.selectbox('Variable Y', numeric_cols, index=min(1, len(numeric_cols)-1))
    covars = st.multiselect('Covariate(s)', [c for c in numeric_cols if c not in {x, y}])
    method = st.selectbox('Method', ['pearson', 'spearman'])
    default_code = f'''result = pg.partial_corr(data=df, x={q(x)}, y={q(y)}, covar={list_code(covars)}, method={q(method)})'''

elif analysis == '5. T-test':
    st.write('One-sample, paired-samples, or independent-samples t-test.')
    require(numeric_cols, 'No numeric variables were found.')
    mode = st.radio('Design', ['One sample', 'Paired samples', 'Independent samples'], horizontal=True)
    alternative = st.selectbox('Alternative hypothesis', ['two-sided', 'greater', 'less'])
    if mode == 'One sample':
        x = st.selectbox('Test variable', numeric_cols)
        mu = st.number_input('Test value', value=0.0)
        default_code = f'''result = pg.ttest(x=df[{q(x)}].dropna(), y={float(mu)}, paired=False, alternative={q(alternative)})'''
    elif mode == 'Paired samples':
        c1, c2 = st.columns(2)
        x = c1.selectbox('Variable 1', numeric_cols)
        y = c2.selectbox('Variable 2', numeric_cols, index=min(1, len(numeric_cols)-1))
        default_code = f'''paired_data = df[[{q(x)}, {q(y)}]].dropna()
result = pg.ttest(x=paired_data[{q(x)}], y=paired_data[{q(y)}], paired=True, alternative={q(alternative)})'''
    else:
        outcome = st.selectbox('Outcome', numeric_cols)
        groups = [c for c in all_cols if c != outcome and df[c].dropna().nunique() >= 2]
        require(groups, 'No grouping variable with at least two levels was found.')
        group = st.selectbox('Grouping variable', groups)
        levels = df[group].dropna().unique().tolist()
        selected = st.multiselect('Choose exactly two groups', levels, default=levels[:2])
        if len(selected) == 2:
            g1, g2 = selected
            default_code = f'''x = df.loc[df[{q(group)}] == {repr(g1)}, {q(outcome)}].dropna()
y = df.loc[df[{q(group)}] == {repr(g2)}, {q(outcome)}].dropna()
result = pg.ttest(x=x, y=y, paired=False, alternative={q(alternative)})'''
        else:
            default_code = '# Choose exactly two groups in the GUI.'

elif analysis == '6. One-way ANOVA':
    st.write('Classical or Welch one-way ANOVA, with optional post hoc tests.')
    require(numeric_cols, 'No numeric dependent variable was found.')
    dv = st.selectbox('Dependent variable', numeric_cols)
    groups = [c for c in all_cols if c != dv and df[c].dropna().nunique() >= 2]
    require(groups, 'No suitable factor was found.')
    between = st.selectbox('Factor', groups)
    method = st.radio('ANOVA type', ['Classical', 'Welch'], horizontal=True)
    posthoc = st.selectbox('Post hoc', ['None', 'Tukey', 'Games–Howell'])
    main = f'result = pg.anova(data=df, dv={q(dv)}, between={q(between)}, detailed=True)' if method == 'Classical' else f'result = pg.welch_anova(data=df, dv={q(dv)}, between={q(between)})'
    extra = ''
    if posthoc == 'Tukey':
        extra = f'\nposthoc = pg.pairwise_tukey(data=df, dv={q(dv)}, between={q(between)})'
    elif posthoc == 'Games–Howell':
        extra = f'\nposthoc = pg.pairwise_gameshowell(data=df, dv={q(dv)}, between={q(between)})'
    default_code = main + extra

elif analysis == '7. Repeated-measures ANOVA':
    st.write('Repeated-measures ANOVA for long-format data.')
    require(numeric_cols, 'No numeric dependent variable was found.')
    dv = st.selectbox('Dependent variable', numeric_cols)
    within = st.selectbox('Within-subject factor', [c for c in all_cols if c != dv])
    subject = st.selectbox('Subject ID', [c for c in all_cols if c not in {dv, within}])
    default_code = f'''result = pg.rm_anova(data=df, dv={q(dv)}, within={q(within)}, subject={q(subject)}, detailed=True, correction="auto")
posthoc = pg.pairwise_tests(data=df, dv={q(dv)}, within={q(within)}, subject={q(subject)}, paired=True, padjust="holm")'''

elif analysis == '8. ANCOVA':
    st.write('ANCOVA with one categorical factor and one or more numeric covariates.')
    require(numeric_cols, 'No numeric dependent variable was found.')
    dv = st.selectbox('Dependent variable', numeric_cols)
    groups = [c for c in all_cols if c != dv and df[c].dropna().nunique() >= 2]
    require(groups, 'No suitable grouping factor was found.')
    between = st.selectbox('Factor', groups)
    covars = st.multiselect('Covariate(s)', [c for c in numeric_cols if c != dv])
    default_code = f'''result = pg.ancova(data=df, dv={q(dv)}, between={q(between)}, covar={list_code(covars)})'''

elif analysis == '9. Nonparametric tests':
    st.write('Mann–Whitney U, Wilcoxon, Kruskal–Wallis, or Friedman test.')
    require(numeric_cols, 'No numeric variables were found.')
    test = st.selectbox('Test', ['Mann–Whitney U', 'Wilcoxon signed-rank', 'Kruskal–Wallis', 'Friedman'])
    if test == 'Mann–Whitney U':
        outcome = st.selectbox('Outcome', numeric_cols)
        groups = [c for c in all_cols if c != outcome and df[c].dropna().nunique() >= 2]
        require(groups, 'No grouping variable was found.')
        group = st.selectbox('Grouping variable', groups)
        levels = df[group].dropna().unique().tolist()
        selected = st.multiselect('Choose exactly two groups', levels, default=levels[:2])
        if len(selected) == 2:
            g1, g2 = selected
            default_code = f'''x = df.loc[df[{q(group)}] == {repr(g1)}, {q(outcome)}].dropna()
y = df.loc[df[{q(group)}] == {repr(g2)}, {q(outcome)}].dropna()
result = pg.mwu(x=x, y=y, alternative="two-sided")'''
        else:
            default_code = '# Choose exactly two groups in the GUI.'
    elif test == 'Wilcoxon signed-rank':
        c1, c2 = st.columns(2)
        x = c1.selectbox('Variable 1', numeric_cols)
        y = c2.selectbox('Variable 2', numeric_cols, index=min(1, len(numeric_cols)-1))
        default_code = f'''paired_data = df[[{q(x)}, {q(y)}]].dropna()
result = pg.wilcoxon(x=paired_data[{q(x)}], y=paired_data[{q(y)}], alternative="two-sided")'''
    elif test == 'Kruskal–Wallis':
        dv = st.selectbox('Dependent variable', numeric_cols)
        between = st.selectbox('Factor', [c for c in all_cols if c != dv])
        default_code = f'''result = pg.kruskal(data=df, dv={q(dv)}, between={q(between)})'''
    else:
        dv = st.selectbox('Dependent variable', numeric_cols)
        within = st.selectbox('Within-subject factor', [c for c in all_cols if c != dv])
        subject = st.selectbox('Subject ID', [c for c in all_cols if c not in {dv, within}])
        default_code = f'''result = pg.friedman(data=df, dv={q(dv)}, within={q(within)}, subject={q(subject)})'''

elif analysis == '10. Linear regression':
    st.write('Simple or multiple ordinary least-squares regression.')
    require(numeric_cols, 'At least two numeric variables are required.')
    outcome = st.selectbox('Outcome', numeric_cols)
    predictors = st.multiselect('Predictor(s)', [c for c in numeric_cols if c != outcome])
    default_code = f'''result = pg.linear_regression(X=df[{list_code(predictors)}], y=df[{q(outcome)}], add_intercept=True, remove_na=True)'''

elif analysis == '11. Reliability':
    st.write("Cronbach's alpha and confidence interval for scale items.")
    require(numeric_cols, 'No numeric scale items were found.')
    items = st.multiselect('Scale items', numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
    default_code = f'''alpha, ci = pg.cronbach_alpha(data=df[{list_code(items)}], nan_policy="pairwise")
result = pd.DataFrame({{"Cronbach alpha": [alpha], "CI lower": [ci[0]], "CI upper": [ci[1]]}})'''

elif analysis == '12. Chi-square test':
    st.write('Chi-square test of independence between two categorical variables.')
    candidates = [c for c in all_cols if df[c].dropna().nunique() >= 2]
    require(candidates, 'At least two variables with multiple categories are required.')
    c1, c2 = st.columns(2)
    x = c1.selectbox('Row variable', candidates)
    y = c2.selectbox('Column variable', [c for c in candidates if c != x])
    default_code = f'''expected, observed, result = pg.chi2_independence(data=df, x={q(x)}, y={q(y)}, correction=True)'''

st.subheader('Editable analysis code')
st.caption('The uploaded dataset is named `df`. Keep the main output in `result`.')
code = st.text_area('Python code', value=default_code, height=230, label_visibility='collapsed')

with st.expander('Code environment'):
    st.code('Available names: df, pd, np, pg, plt, sns')
    st.warning('The editable runner is for a trusted classroom workspace. It is not a security sandbox.')

if st.button('Run analysis', type='primary'):
    if not code.strip():
        st.warning('There is no code to run.')
        st.stop()
    namespace = {'df': df.copy(), 'pd': pd, 'np': np, 'pg': pg, 'plt': plt, 'sns': sns}
    safe_builtins = {'print': print, 'len': len, 'range': range, 'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round, 'list': list, 'dict': dict, 'tuple': tuple, 'float': float, 'int': int, 'str': str, 'bool': bool}
    console = io.StringIO()
    try:
        plt.close('all')
        with contextlib.redirect_stdout(console):
            exec(code, {'__builtins__': safe_builtins}, namespace)
        if console.getvalue():
            st.subheader('Console')
            st.code(console.getvalue())
        st.subheader('Results')
        found = False
        for name in ['result', 'posthoc', 'observed', 'expected']:
            if name not in namespace:
                continue
            found = True
            value = namespace[name]
            st.markdown(f'**{name}**')
            if isinstance(value, pd.DataFrame):
                st.dataframe(value, use_container_width=True)
                st.download_button(f'Download {name}.csv', value.to_csv(index=True).encode('utf-8'), file_name=f'{name}.csv', mime='text/csv', key=f'download_{name}')
            elif isinstance(value, pd.Series):
                st.dataframe(value.to_frame(), use_container_width=True)
            else:
                st.write(value)
        for number in plt.get_fignums():
            found = True
            st.pyplot(plt.figure(number), clear_figure=False)
        if not found:
            st.info('The code ran, but no `result`, `posthoc`, `observed`, `expected`, or plot was created.')
    except Exception:
        st.error('The analysis could not be completed.')
        st.code(traceback.format_exc())

with st.expander('Interpretation reminders'):
    st.markdown('''
- Select analyses based on the research design, not only on the p-value.
- Inspect missing data, outliers, distributions, and group sizes.
- Repeated-measures and Friedman analyses require long-format data.
- Independent t-tests and Mann–Whitney tests require exactly two independent groups.
- Correlation does not establish causation.
- Compare important results with another statistical package when accuracy is critical.
''')
