from .models import RegData, ColumnTypes, DataOutput, FileMetaData
import pandas as pd
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_regression
from pandas.api.types import is_numeric_dtype
import plotly.graph_objects as go
import plotly.offline as opy
import plotly.io as pio
import networkx as nx
from collections import defaultdict
from plotly.subplots import make_subplots
import time
import math

pio.templates["plotly_white_custom"] = pio.templates["plotly_white"]
pio.templates["plotly_white_custom"]["layout"]["title_font_size"] = 20
pio.templates["plotly_white_custom"]["layout"]["font_family"] = "Gravitas One"
pio.templates["plotly_white_custom"]["layout"]["font_color"] = "#2D3949"
pio.templates["plotly_white_custom"]["layout"]["xaxis"]["tickfont"] = {"size": 14}
pio.templates["plotly_white_custom"]["layout"]["yaxis"]["tickfont"] = {"size": 14}
pio.templates["plotly_white_custom"]["layout"]["xaxis"]["title_font"] = {"size": 15}
pio.templates["plotly_white_custom"]["layout"]["yaxis"]["title_font"] = {"size": 15}


def set_up_buttons(buttons):
    updatemenu = []
    your_menu = dict()
    updatemenu.append(your_menu)
    updatemenu[0]['buttons'] = buttons
    updatemenu[0]['direction'] = 'down'
    updatemenu[0]['x'] = 0.05
    updatemenu[0]['y'] = 1.15
    updatemenu[0]['showactive'] = True
    return updatemenu


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
        self.normalised_df = None
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
        self.normalised_df = (self.df - self.df.mean()) / self.df.std()

    def save_corr_matrix(self):
        corr = self.df[self.col_types['int']].corr().reset_index()
        # checking if exists
        existing = DataOutput.objects.filter(project_name=self.title, output_name='corr_matrix').exists()
        if existing:
            DataOutput.objects.filter(project_name=self.title, output_name='corr_matrix').delete()
        # saving corr matrix to plot in java script
        DataOutput(output=pd.melt(corr, id_vars='index').to_dict(orient='records'), output_name='corr_matrix',
                   project_name=FileMetaData.objects.get(project_name=self.title)).save()
        corr_new = corr.set_index('index')
        fig = go.Figure(data=go.Heatmap(
            x=corr_new.columns,
            y=corr_new.index,
            z=corr_new,
            colorscale=[(0, "#ff9900"), (0.5, 'white'), (1, "#2D3949")]))
        fig.update_layout(showlegend=False, title_text=f"Feature Correlation Matrix",
                          template="plotly_white_custom")
        fig.update_yaxes(title=None)
        fig.update_xaxes(tickangle=45)
        return corr, opy.plot(fig, auto_open=False, output_type='div')

    def plot_distributions(self):
        df = self.normalised_df[self.col_types['int']].set_index(self.y_cols[0])
        if df.shape[0] > 1000:
            df = df.sample(1000)
        fig = go.Figure()

        def get_kde(col, var):
            fig_temp = ff.create_distplot([df[col]], ['distplot'])  # TODO: check curve_type='normal'
            x = fig_temp['data'][1].x
            y = fig_temp['data'][1].y
            if var == 'x':
                return x
            elif var == 'y':
                return y

        fig_temp = ff.create_distplot([df[df.columns[0]]], ['distplot'])
        fig.add_trace(go.Histogram(x=df[df.columns[0]], visible=True, histnorm='probability density',
                                   marker_color='#2D3949', opacity=0.75))
        fig.add_trace(go.Scatter(x=fig_temp['data'][1].x, y=fig_temp['data'][1].y, visible=True,
                                 marker_color='#ff9900'))
        buttons = []
        for col in df.columns:
            buttons.append(dict(method='restyle',
                                label=col,
                                visible=True,
                                args=[{'y': [None, get_kde(col, 'y')],
                                       'x': [df[col], get_kde(col, 'x')],
                                       'type': ['histogram', 'scatter']}],
                                )
                           )

        updatemenu = set_up_buttons(buttons)
        fig.update_layout(showlegend=False, updatemenus=updatemenu,
                          title_text=f"Distribution",
                          template="plotly_white_custom",
                          )

        return opy.plot(fig, auto_open=False, output_type='div')

    def plot_xy_linearity(self):
        df = self.normalised_df[self.col_types['int']].set_index(self.y_cols[0])
        if df.shape[0] > 1000:
            df = df.sample(1000)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[df.columns[0]],
                                 y=df.index,
                                 visible=True,
                                 mode='markers')
                      )
        fig.update_yaxes(title_text=self.y_cols[0])
        buttons = []
        for col in df.columns:
            buttons.append(dict(method='restyle',
                                label=col,
                                visible=True,
                                args=[{'y': [df.index],
                                       'x': [df[col]],
                                       'type': 'scatter'}, [0]],
                                )
                           )

        updatemenu = set_up_buttons(buttons)
        fig.update_layout(showlegend=False, updatemenus=updatemenu,
                          title_text=f"Scatter plot of X & Y",
                          template="plotly_white_custom"
                          )
        fig.update_traces(marker=dict(size=5,
                                      line=dict(width=2,
                                                color='#2D3949')),
                          selector=dict(mode='markers')
                          )
        return opy.plot(fig, auto_open=False, output_type='div')

    def plot_categorical_data(self):
        # TODO: add to the front-end
        # TODO: add variance number
        bar_plot_cols = []
        for col in self.col_types['c']:
            if self.df[col].nuinique() < 10:
                bar_plot_cols.append(col)

        df = self.df[bar_plot_cols]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df[df.columns[0]].value_counts().index.to_list(),
                             y=df[df.columns[0]].value_counts().to_list(),
                             visible=True)
                      )
        buttons = []
        for col in df.columns:
            buttons.append(dict(method='restyle',
                                label=col,
                                visible=True,
                                args=[{'y': [df[col].value_counts().to_list()],
                                       'x': [df[col].value_counts().index.to_list()],
                                       'type': 'bar'}, [0]],
                                )
                           )

        updatemenu = set_up_buttons(buttons)
        fig.update_layout(showlegend=False, updatemenus=updatemenu,
                          title_text=f"Categorical Features - Bar Chart",
                          template="plotly_white"
                          )
        fig.update_traces(marker_color='#2D3949', marker_line_color='#000000',
                          marker_line_width=1.5, opacity=0.7)
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
                          template="plotly_white_custom")
        fig.update_xaxes(tickangle=45)
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
        # TODO: remove columns with low variance and a large number of unique values
        # TODO: f_scores
        return cols_to_remove

    def run(self):
        corr_matrix, div_corr = self.save_corr_matrix()
        div_lin = self.plot_xy_linearity()
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
        self.results = defaultdict()

    def split_train_test(self, df, y_col, drop_cols, numeric_features, categorical_features):
        df = df.drop(drop_cols, axis=1)
        X, y = df.drop(y_col, axis=1), df[y_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=42, test_size=0.2)
        self.numeric_cols = list(set(numeric_features).intersection(X.columns))
        self.categorical_cols = list(set(categorical_features).intersection(X.columns))

    def transform_cols(self):
        #Normalizing the output will not affect shape of 𝑓, so it's generally not necessary.
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = make_column_transformer(
            (numeric_transformer, self.numeric_cols),
            (categorical_transformer, self.categorical_cols),
            remainder='passthrough')
        return preprocessor

    def train_pipeline(self, preprocessor):
        rf_pipeline = make_pipeline(preprocessor,
                                    RandomForestRegressor(random_state=42, n_estimators=50))
        gradient_pipeline = make_pipeline(
            preprocessor,
            GradientBoostingRegressor(random_state=0))
        regressor = make_pipeline(preprocessor,
                                  LinearRegression())
        ridge_reg = RidgeCV([1e-3, 1e-2, 1e-1, 1])
        poly_reg = PolynomialFeatures(degree=2, include_bias=False)
        poly_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ("poly_features", poly_reg),
            ('regul_reg', ridge_reg)])
        return [regressor, rf_pipeline, gradient_pipeline, poly_pipeline]

    def plot_model_performance(self):
        fig = make_subplots(rows=2, cols=2, subplot_titles=tuple(self.results.keys()))
        pos = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for i, res in enumerate(self.results.items()):
            row, col = pos[i]
            train_score, test_score, mse, rmse, y_pred, elapsed_time = res[1].values()
            y_true = self.y_test
            y_pred = y_pred
            # scatter plot
            fig.add_trace(
                go.Scatter(x=y_true, y=y_pred, mode='markers', marker_color='#2D3949'),
                row=row, col=col
            )
            # trend line
            fig.add_trace(
                go.Scatter(x=[y_true.min(), y_true.max()],
                           y=[y_true.min(), y_true.max()],
                           line=dict(color='#eb6600', width=2, dash='dot')), row=row, col=col,
            )
            
            # box
            fig.add_shape(
                type="rect",
                xref="x domain", yref="y domain",
                x0=0.05, x1=0.3, y0=0.5, y1=0.95,
                opacity=0.5,
                line=dict(
                    color="#ff9900",
                    width=1
                ),
                fillcolor="#ffd89d",
                row=row,
                col=col,
                layer='below'
            )
            
            # text
            fig.add_annotation(font=dict(size=11),
                               xref="x domain",
                               yref="y domain",
                               x=0.07, y=0.95,
                               text=f'R^2 train:{train_score:9.2f}<br>R^2 test:{test_score:9.2f}<br>RMSE:{rmse:9.2f}',
                               row=row,
                               col=col,
                               showarrow=False,
                               align='left'
                               )

        fig.update_layout(
            showlegend=False,
            template="plotly_white_custom"
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        fig.update_layout(title_text="Model Training - Regression Results")
        return opy.plot(fig, auto_open=False, output_type='div')

    def run(self):
        preprocessor = self.transform_cols()
        pipelines = self.train_pipeline(preprocessor)
        for estimator in pipelines:
            start_time = time.time()
            estimator.fit(self.X_train, self.y_train)
            y_pred = estimator.predict(self.X_test)
            name = estimator[-1].__class__.__name__
            train_score = estimator.score(self.X_train, self.y_train)  # Returns the coeffof determination R2 of the prediction.
            test_score = estimator.score(self.X_test, self.y_test)  # r2
            mse = mean_squared_error(self.y_test, y_pred)
            elapsed_time = time.time() - start_time
            # R2 is a normalized version of MSE
            self.results[name] = {'train_r2': train_score, 'test_r2': test_score,
                                  'mse': mse, 'rmse': math.sqrt(mse),
                                  'y_pred': y_pred, 'elapsed_time': elapsed_time}

