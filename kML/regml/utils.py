import pandas as pd
from .models import RegData, FileMetaData


class DataFrameImputer():

    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        """
        self.df = None

    def res(self, X):
        self.df = X.copy()
        for col in X:
            if X[col].dtype == float or X[col].dtype == int:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])
        return self


def handle_uploaded_file(f, y, tick):
    y = y.lower()
    df = pd.read_csv(f)
    df.columns = [x.lower() for x in df.columns]
    # TODO: if more than 80% of data are null then drop column
    if tick:
        df.dropna(inplace=True)
    else:
        df = DataFrameImputer().res(df).df
    cols = list(df.columns)
    cols.remove(y)
    d_y = df[y].to_dict()
    d_x = df[cols].to_dict('split')
    return d_x, d_y


def load_file_into_db(x, y, title):
    l = []
    for col1, col2 in zip(x, list(y.values())):
        l.append(RegData(x=col1, y=col2,
                         project_name=FileMetaData.objects.get(project_name=title)))
    RegData.objects.bulk_create(l)


def load_file_metadata(x_colnames, y_colname, title):
    FileMetaData(col_names=x_colnames,
                 y_name=y_colname,
                 project_name=title).save()
