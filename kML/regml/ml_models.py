from .models import RegData, ColumnTypes, DataOutput, FileMetaData, Dropdown
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
from pandas.api.types import is_numeric_dtype
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as opy
import plotly.express as px
import math
import networkx as nx


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
        self.build_df()

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
        self.retrieve_columns()
        self.retrieve_observations()
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
        existing = Dropdown.objects.filter(project_name=self.title).exists()
        if existing:
            Dropdown.objects.filter(project_name=self.title).delete()
        for c in self.col_types['int']:
            Dropdown(col_name=c, project_name=FileMetaData.objects.get(project_name=self.title)).save()

    def save_corr_matrix(self):
        corr = self.df[self.col_types['int']].corr().reset_index()
        # checking if exists
        existing = DataOutput.objects.filter(project_name=self.title, output_name='corr_matrix').exists()
        if existing:
            DataOutput.objects.filter(project_name=self.title, output_name='corr_matrix').delete()
        # saving corr matrix to plot in java script
        DataOutput(output=pd.melt(corr, id_vars='index').to_dict(orient='records'), output_name='corr_matrix',
                   project_name=FileMetaData.objects.get(project_name=self.title)).save()
        fig = px.imshow(corr.set_index('index'),
                        color_continuous_scale=[(0, "#ff9900"), (0.5, 'white'), (1, "#2D3949")],
                        )
        fig.update_layout(showlegend=False, title_text=f"Feature Correlation Matrix",
                          template="presentation")
        fig.update_yaxes(title=None)
        fig.update_xaxes(tickangle=45)
        fig.update_layout(
            font_family="Gravitas One",
            font_color="#2D3949",
        )
        return corr, opy.plot(fig, auto_open=False, output_type='div')

    def plot_xy_linearity(self, var):
        extract_df = self.df[[var, self.y_cols[0]]]
        normalized_df = (extract_df - extract_df.mean()) / extract_df.std()
        if normalized_df.shape[0] > 1000:
            normalized_df = normalized_df.sample(1000)
        target = self.y_cols[0]
        fig = px.scatter(normalized_df, x=var, y=target)
        fig.update_layout(showlegend=False, title_text=f"{var.capitalize()} - Target {target.capitalize()}",
                          template="presentation")
        fig.update_traces(marker=dict(size=5,
                                      line=dict(width=2,
                                                color='#2D3949')),
                          selector=dict(mode='markers'))
        fig.update_layout(
            font_family="Gravitas One",
            font_color="#2D3949",
        )
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
        fig.update_traces(marker_color="#ff9900", marker_line_color='#2D3949',
                          marker_line_width=1.5, opacity=0.8)
        fig.update_layout(showlegend=False, title_text=f"Feature F-scores",
                          template="presentation")
        fig.update_xaxes(tickangle=45)
        fig.update_layout(
            font_family="Gravitas One",
            font_color="#2D3949",
        )
        return opy.plot(fig, auto_open=False, output_type='div')

    def propose_columns_to_remove(self, corr_matrix, corr_thresh=0.6):
        """
        Columns have to fall into the below to be removed:
            - highly correlated
            - have no linear relationship with y
            - categorical columns that have too many unique values
            - high p-value of the f-scores is not significant (>0.05)
        """
        cols_to_remove = []
        # corr part
        corr_matrix = pd.melt(corr_matrix, id_vars='index')
        corr_matrix['value'] = corr_matrix['value'].abs()
        corr_matrix = corr_matrix[~(corr_matrix['index'] == corr_matrix['variable'])]
        corr_matrix_no_y = corr_matrix[~((corr_matrix['index'] == self.y_cols[0]) |
                                       (corr_matrix['variable'] == self.y_cols[0]))]
        high_corr_pair = list(corr_matrix_no_y[corr_matrix_no_y.value > corr_thresh][['index', 'variable']]\
                             .itertuples(index=False, name=None))
        if len(high_corr_pair) > 0:
            unique_pairs = set(tuple(sorted(l)) for l in high_corr_pair)
            G = nx.Graph()
            G.add_edges_from(unique_pairs)
            groups = list(nx.connected_components(G))
            scores = nx.betweenness_centrality(G)
            for g in groups:
                if len(g) == 2:
                    rm_val = corr_matrix[(corr_matrix['index'] == self.y_cols[0]) &
                                (corr_matrix['variable'].isin(g))]\
                        .sort_values(by='value').iloc[0].variable
                    cols_to_remove.append(rm_val)
                else:
                    g_scores = {k: v for k, v in scores.items() if k in g}
                    g.remove(max(g_scores, key=g_scores.get))
                    cols_to_remove.extend(list(g))
        # date columns
        cols_to_remove.extend(self.col_types['d'])
        # TODO: categorical
        # TODO: f_scores
        return cols_to_remove

    def run(self, var):
        corr_matrix, div_corr = self.save_corr_matrix()
        div_lin = self.plot_xy_linearity(var)
        div_f = self.calculate_f_scores()
        cols_to_remove = self.propose_columns_to_remove(corr_matrix)
        return div_corr, div_lin, div_f, cols_to_remove


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




