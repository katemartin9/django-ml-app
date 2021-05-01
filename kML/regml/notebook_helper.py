from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
import plotly.graph_objects as go


class CorrMatrix:

    def __init__(self):
        self.corr = None
        self.fig = None

    def calculate(self, df):
        self.corr = df.corr()

    def plot(self, template="plotly_white"):
        fig = go.Figure(data=go.Heatmap(
            x=self.corr.columns,
            y=self.corr.index,
            z=self.corr,
            colorscale=[(0, "#ff9900"), (0.5, 'white'), (1, "#2D3949")]))
        fig.update_layout(showlegend=False, title_text=f"Feature Correlation Matrix",
                          template=template)
        fig.update_yaxes(title=None)
        fig.update_xaxes(tickangle=45)
        self.fig = fig


class DataScalerCl:

    def __init__(self, scaler='standard'):
        self.scaler = dict(
            standard=StandardScaler(),
            minmax=MinMaxScaler(),
            robust=RobustScaler(),
            onehot=OneHotEncoder()
        )
        self.spec = scaler

    def call(self, df):
        cl = self.scaler[self.spec]
        cl.fit(df)


class FeatureSelectionCl:

    def __init__(self, df, y):
        self.df = df
        self.x = None
        self.y = y
        self.col_to_remove_ = set()

    def encoding_catvars(self):
        # Encodes target labels with value between 0 and n_classes-1
        le = LabelEncoder()
        x_enc = self.df.drop(self.y, axis=1).copy()
        for col in list(x_enc.select_dtypes(include=['object']).columns):
            x_enc[col] = le.fit_transform(x_enc[col])
        self.x = x_enc

    def variance_selection(self):
        # Default threshold = 0 -> same value in every sample of that feature
        selector = VarianceThreshold(threshold=0.2)
        selector.fit_transform(self.x)
        cols_to_remove = self.x.columns[~selector.get_support()]
        self.col_to_remove_.update(cols_to_remove)

    def f_scores_selection(self, alpha=0.05):
        y = self.df[self.y]
        f_val, p_val = f_regression(self.x, y)
        cols_to_remove = self.x.columns[p_val > alpha]
        self.col_to_remove_.update(cols_to_remove)

    def fit(self):
        self.encoding_catvars()
        self.variance_selection()
        self.f_scores_selection()