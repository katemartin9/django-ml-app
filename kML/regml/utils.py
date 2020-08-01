import pandas as pd
from .models import RegData
import numpy as np
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        """
        self.fill = None

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def handle_uploaded_file(f, y, tick):
    y = y.lower()
    df = pd.read_csv(f)
    df.columns = [x.lower() for x in df.columns]
    if tick:
        df.dropna(inplace=True)
    else:
        DataFrameImputer().fit_transform(df)
    cols = list(df.columns)
    cols.remove(y)
    d_y = df[y].to_dict()
    d_x = df[cols].to_dict('split')
    return d_x, d_y


def load_file_into_db(x, y, title):
    l = []
    for col1, col2 in zip(x['data'], list(y.values())):
        l.append(RegData(x=col1, y=col2, filename=title))
    RegData.objects.bulk_create(l)
