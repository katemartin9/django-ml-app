from .models import RegData, ColumnTypes, DataOutput, FileMetaData
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
import numpy as np
from .utils import Container, plot_regression_results
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as opy
import math


class FeatureSelection:

    def __init__(self, title):
        self.title = title
        self.data_dict = {}
        self.x_cols = []
        self.y_cols = []
        self.col_types = {
            'n': [],
            'c': [],
            'd': []
            }
        self.df = None

    def retrieve_columns(self):
        for row in ColumnTypes.objects.all().filter(project_name=self.title):
            if row.y:
                self.y_cols.append(row.col_name)
                self.col_types[row.col_type].append(row.col_name)
            else:
                self.x_cols.append(row.col_name)
                self.col_types[row.col_type].append(row.col_name)

    def retrieve_observations(self):
        for i, row in enumerate(RegData.objects.all().filter(project_name=self.title)):
            self.data_dict[i] = row.observations

    def build_df(self):
        self.df = pd.DataFrame.from_dict(self.data_dict, orient='index')
        # check all numeric columns have been converted
        for col in self.col_types['n']:
            if not is_numeric_dtype(self.df[col]):
                self.df[col] = pd.to_numeric(self.df[col])
        # TODO: check all categorical columns have been converted

    def save_corr_matrix(self):
        corr = self.df[self.col_types['n']].corr().reset_index()
        # checking if exists
        existing = DataOutput.objects.filter(project_name=self.title, output_name='corr_matrix').exists()
        if existing:
            DataOutput.objects.filter(project_name=self.title, output_name='corr_matrix').delete()
        # saving corr matrix to plot in java script
        DataOutput(output=pd.melt(corr, id_vars='index').to_dict(orient='records'), output_name='corr_matrix',
                   project_name=FileMetaData.objects.get(project_name=self.title)).save()

    def plot_xy_linearity(self):
        normalized_df = (self.df[self.col_types['n']] -
                         self.df[self.col_types['n']].mean()) / self.df[self.col_types['n']].std()
        if normalized_df.shape[0] > 500:
            normalized_df = normalized_df.sample(500)
        x = self.y_cols[0]
        temp_cols = list(normalized_df.columns)
        temp_cols.remove(x)
        len(temp_cols)

        fig = make_subplots(rows=math.ceil(len(temp_cols) / 2), cols=2, start_cell="bottom-left",
                            subplot_titles=tuple(temp_cols))
        rows = 0
        for i, y in enumerate(temp_cols):
            if (i + 1) % 2 == 0:
                cols = 2
            else:
                cols = 1
                rows += 1
            fig.add_trace(go.Scatter(x=normalized_df[x], y=normalized_df[y], mode='markers'), row=rows, col=cols)
        fig.update_layout(showlegend=False, title_text=f"Linear Relationship of {x} (x axis) and features (y axis)",
                          template="plotly_white")
        fig.update_layout(
            autosize=False,
            width=900,
            # TODO:better adjust height
            height=1000)
        return opy.plot(fig, auto_open=False, output_type='div')

    def run(self):
        self.retrieve_columns()
        self.retrieve_observations()
        self.build_df()
        self.save_corr_matrix()
        div = self.plot_xy_linearity()
        return div


# TODO: integrate into the above code some remaining logic and delete
"""
# feature selection based on corr coefficients
corr_matrix = pd.melt(df.corr().reset_index(), id_vars='index')
corr_matrix = corr_matrix[corr_matrix.value < 1.0]
high_pos_corr = list(corr_matrix[corr_matrix.value > 0.8][['index', 'variable']]\
    .itertuples(index=False, name=None))
corr_with_y = corr_matrix[(corr_matrix.variable == y_col[0]) &
                          ((corr_matrix.value > 0.4) | (corr_matrix.value < -0.4))]['index'].to_list()
features = [Container(x) for x in high_pos_corr]
#columns_to_drop.extend(highly_corr)
#columns_to_drop.extend(['TAX', 'RAD', 'AGE'])




# STEP 4: add categorical columns
if len(col_types['categorical']) > 0:
    for col in col_types['categorical']:
        vals = df[col].nunique()
        if vals < 10:
            categorical_features.append(col)
        else:
            columns_to_drop.append(col)

# STEP 5: drop columns that are not relevant
# drop dates
if len(col_types['dates']) > 0:
    columns_to_drop.extend(col_types['dates'])

# drop selected columns
df.drop(columns_to_drop, axis=1, inplace=True)

# STEP 6: Train test split
X, y = df.drop(y_name, axis=1), df[y_name].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# STEP 7: BUILD PIPELINE
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = make_column_transformer(
        (numeric_transformer, list(set(numeric_features).intersection(df.columns))),
        (categorical_transformer, list(set(categorical_features).intersection(df.columns))),
    remainder='passthrough')

# STEP 8: RUN THROUGH ESTIMATORS
rf_pipeline = make_pipeline(preprocessor,
                            RandomForestRegressor(random_state=42, n_estimators=50))
gradient_pipeline = make_pipeline(
    preprocessor,
    HistGradientBoostingRegressor(random_state=0))
regressor = make_pipeline(preprocessor,
                          LinearRegression())
ridge_reg = RidgeCV([1e-3, 1e-2, 1e-1, 1])
poly_reg = PolynomialFeatures(degree=2, include_bias=False)
poly_pipeline = Pipeline([
    ("poly_features", poly_reg),
    ("std_scaler", numeric_transformer),
    ('regul_reg', ridge_reg)])

# STEP 9: PLOT THE RESULTS
fig, axs = plt.subplots(2, 2, figsize=(9, 7))
axs = np.ravel(axs)
for ax, estimator in zip(axs, [rf_pipeline, regressor, gradient_pipeline, poly_pipeline]):
    start_time = time.time()
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    name = estimator[-1].__class__.__name__
    train_score = estimator.score(X_train, y_train)  # r2
    test_score = estimator.score(X_test, y_test)  # r2
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    elapsed_time = time.time() - start_time
    plot_regression_results(ax, y_test, y_pred,
                            (r'$R^2 train={:.2f}' + '\n' + '$R^2 test={:.2f}' + '\n' + r'$RMSE={:.2f}')
                            .format(train_score, test_score, rmse), name, elapsed_time),
plt.suptitle('Single predictors versus stacked predictors', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# STEP 10: OUTPUT RESULT

"""


