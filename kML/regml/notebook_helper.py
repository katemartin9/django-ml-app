from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_regression


class CorrMatrix:
    def __init__(self):
        pass

    def calculate(self):
        pass

    def plot(self):
        pass


class FeatureSelectionCl:

    def __init__(self, df, y):
        self.df = df
        self.y = y
        self.col_to_remove_ = []

    def encoding_catvars(self):
        # Encodes target labels with value between 0 and n_classes-1
        le = LabelEncoder()
        x_enc = self.df.drop(self.y, axis=1).copy()
        for col in list(x_enc.select_dtypes(include=['object']).columns):
            x_enc[col] = le.fit_transform(x_enc[col])
        self.df = x_enc

    def variance_selection(self):
        # Default threshold = 0 -> same value in every sample of that feature
        selector = VarianceThreshold(threshold=0.2)
        cols_to_remove = self.df.columns[selector.get_support()]
        self.col_to_remove_.extend(cols_to_remove)

    def f_scores_selection(self, alpha=0.05):
        y = self.df[self.y]
        f_val, p_val = f_regression(self.df, y)
        cols_to_remove = self.df.columns[p_val > alpha]
        self.col_to_remove_.extend(cols_to_remove)

    def run(self):
        self.encoding_catvars()
        self.variance_selection()
        self.f_scores_selection()