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
        # TODO: check all numeric columns have been converted
        for col in self.col_types['n']:
            if not is_numeric_dtype(self.df[col]):
                self.df[col] = pd.to_numeric(self.df[col])
        # TODO: check all categorical columns have been converted
        # TODO: deal with the date columns

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

    def train_linreg(self):
        pass

    def run(self):
        self.retrieve_columns()
        self.retrieve_observations()
        self.build_df()
        self.save_corr_matrix()
        div = self.plot_xy_linearity()
        div2 = self.calculate_f_scores()
        return [div, div2]


# TODO: integrate into the above code some remaining logic and delete



