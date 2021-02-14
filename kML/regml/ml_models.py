from .models import RegData, ColumnTypes, DataOutput, FileMetaData
import pandas as pd
import seaborn as sns
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
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, f_regression
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
            'd': [],
            'int': []
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
        for col in self.col_types['n']:
            if not is_numeric_dtype(self.df[col]):
                try:
                    self.df[col] = pd.to_numeric(self.df[col])
                except ValueError as e:
                    self.col_types['n'].remove(col)
                    self.col_types['c'].append(col)

        for col in self.col_types['c']:
            try:
                self.df[col] = pd.to_numeric(self.df[col])
            except ValueError as e:
                pass
            else:
                self.col_types['int'].append(col)
        self.col_types['int'].extend(self.col_types['n'])

    def save_corr_matrix(self):
        corr = self.df[self.col_types['int']].corr().reset_index()
        # checking if exists
        existing = DataOutput.objects.filter(project_name=self.title, output_name='corr_matrix').exists()
        if existing:
            DataOutput.objects.filter(project_name=self.title, output_name='corr_matrix').delete()
        # saving corr matrix to plot in java script
        DataOutput(output=pd.melt(corr, id_vars='index').to_dict(orient='records'), output_name='corr_matrix',
                   project_name=FileMetaData.objects.get(project_name=self.title)).save()

    def plot_xy_linearity(self):
        normalized_df = (self.df[self.col_types['int']] -
                         self.df[self.col_types['int']].mean()) / self.df[self.col_types['int']].std()
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
        return opy.plot(fig, auto_open=False, output_type='div')

    def calculate_f_scores(self):
        label_encoder = LabelEncoder()
        for col in self.col_types['c']:
            self.df[col] = label_encoder.fit_transform(self.df[col])
        new_l = self.col_types['n'][:]
        new_l.extend(self.col_types['c'])
        y = self.y_cols[0]
        new_l.remove(y)
        X = self.df[new_l]
        y = self.df[y]
        f_scores = f_regression(X, y, center=True)
        p_values = pd.Series(f_scores[1], index=X.columns) \
            .sort_values(ascending=False)

        fig = go.Figure([go.Bar(x=p_values.index, y=p_values.values)])
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.6)
        fig.update_layout(showlegend=False, title_text=f"F-scores - Categorical and Numeric Features",
                          template="plotly_white")
        return opy.plot(fig, auto_open=False, output_type='div')

    def run(self):
        self.retrieve_columns()
        self.retrieve_observations()
        self.build_df()
        self.save_corr_matrix()
        div = self.plot_xy_linearity()
        div2 = self.calculate_f_scores()
        return [div, div2]


class RegModel:

    def __init__(self):
        self.categorical_cols = None
        self.numeric_cols = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_train_test(self, df, y_col, drop_cols, numeric_features, categorical_features):
        df = df.drop(drop_cols, axis=1)
        X, y = df.drop(y_col, axis=1), df[y_col].astype(int)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=42, test_size=0.2)
        self.numeric_cols = list(set(numeric_features).intersection(df.columns))
        self.categorical_cols = list(set(categorical_features).intersection(df.columns))

    def transform_cols(self):
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = make_column_transformer(
            (numeric_transformer, self.numeric_cols),
            (categorical_transformer, self.categorical_cols),
            remainder='passthrough')
        return preprocessor, numeric_transformer

    def train_pipeline(self, preprocessor, numeric_transformer):
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
        return rf_pipeline, gradient_pipeline, regressor, poly_pipeline

    def plot_model_performance(self):
        pass

    def run(self):
        self.split_train_test()
        preprocessor, numeric_transformer = self.transform_cols()
        self.train_pipeline(preprocessor, numeric_transformer)




