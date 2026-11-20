# Student Statistics Lab — Teaching Dataset Edition

Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Built-in datasets

- `general_data.csv` — descriptives, normality, correlations, t-tests, ANOVA, ANCOVA, regression
- `repeated_measures_data.csv` — repeated-measures ANOVA and Friedman
- `reliability_data.csv` — Cronbach's alpha with eight questionnaire items
- `categorical_data.csv` — chi-square tests
- `nonparametric_data.csv` — Mann–Whitney, Kruskal–Wallis, and Wilcoxon

Students may also upload CSV, Excel, or SPSS `.sav` files.

Uploads are copied into `Uploads/Datasets` or `Uploads/Codes` in the app directory. The DataFrame CSV and analysis Python downloads are also saved there when their download buttons are clicked. The browser's own download location remains controlled by browser settings.

## Shared SQLite dataset

The sidebar can load a shared dataset from SQLite. Save the current dataset to make it available to all app users; each save replaces the shared copy. When a saved copy exists, it is selected automatically on app load. The default database path is `Uploads/Datasets/shared_dataset.sqlite3`, and `SPYSS_SQLITE_PATH` can be set to choose another path.

The database file must be on persistent, writable storage to retain data across restarts or deployments. Do not assume the app's deployed content directory provides this. SQLite stores one shared dataset, so users can overwrite one another's changes; it does not provide per-user access control.

The code editor also supports a shared list of named `.py` files in this database. Use the shared Python files checkbox to browse that list, and save the current editor contents under a filename; saving the same name replaces that shared file. Shared scripts are visible to all app users, and running one executes its code in the app process, so only share trusted scripts.
