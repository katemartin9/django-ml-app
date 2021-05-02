import pandas as pd
from .models import RegData, FileMetaData, ColumnTypes
import plotly.graph_objects as go


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


class DataFrameImputer:

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        """
        self.df = None

    def res(self, X, threshold=0.6):
        cols_to_keep = list(X.columns[(X.isnull().sum() / X.shape[0]) < threshold])
        X = X[cols_to_keep]
        self.df = X.copy()
        for col in X:
            if X[col].dtype == float or X[col].dtype == int:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])
        return self


class PlotTemplate:

    def __init__(self):
        self.df = None
        self.fig_ = None

    def initialize_figure(self):
        self.fig_ = go.Figure()

    def initialize_scatter(self, col):
        d = dict(x='self.df[col]',
                 y='self.df.index'
                 )
        self.fig_.add_trace(go.Scatter(x=eval(d['x']),
                                       y=eval(d['y']),
                                       visible=True,
                                       mode='markers'))
        return d

    def initialize_bar(self, col):
        d = dict(x='self.df[col].value_counts().index.to_list()',
                 y='self.df[col].value_counts().to_list()'
                 )
        self.fig_.add_trace(go.Bar(x=eval(d['x']),
                                   y=eval(d['y']),
                                   visible=True))
        return d

    def build_dropdown(self, d, t):
        buttons = []
        for col in self.df.columns:
            buttons.append(dict(method='restyle',
                                label=col,
                                visible=True,
                                args=[{'y': [eval(d['y'])],
                                       'x': [eval(d['x'])],
                                       'type': t}, [0]],
                                )
                           )

        updatemenu = set_up_buttons(buttons)
        self.fig_.update_layout(showlegend=False,
                                updatemenus=updatemenu,
                                template="plotly_white")

    def __call__(self, df, plot_type):
        self.initialize_figure()
        self.df = df
        dd_default = self.df.columns[0]
        params = {
            'scatter': self.initialize_scatter,
            'bar': self.initialize_bar,
        }
        plot_init = params[plot_type]
        d = plot_init(dd_default)
        self.build_dropdown(d, plot_type)


def handle_uploaded_file(f, tick):
    df = pd.read_csv(f)
    df.columns = [x.lower() for x in df.columns]
    if tick:
        df.dropna(inplace=True)
    else:
        df = DataFrameImputer().res(df).df
    return df.to_dict(orient='records')


def db_load_file(data, title, user):
    iter_data = []
    FileMetaData(project_name=title, user=user).save()
    for row in data:
        iter_data.append(RegData(observations=row,
                                 project_name=FileMetaData.objects.get(project_name=
                                                                       title)))
    RegData.objects.bulk_create(iter_data)


def db_load_column_types(data, title):
    iter_data = []
    for row in data:
        iter_data.append(ColumnTypes(col_name=row['col_name'],
                                     col_type=row['col_type'],
                                     y=row['y'],
                                     project_name=FileMetaData.objects.get(project_name=
                                                                           title)
                                     ))
    ColumnTypes.objects.bulk_create(iter_data)

