import pandas as pd
from .models import RegData, FileMetaData
import matplotlib.pyplot as plt


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


class Container:

    def __init__(self, *args):
        self.left = args[0][0]
        self.right = args[0][1]

    def __eq__(self, other):
        return set([self.left, self.right]) == set([other.left, other.right])

    def __contains__(self, item):
        return self.left == item or self.right == item

    def __hash__(self):
        return hash(self.left) + hash(self.right)

    def __repr__(self):
        return f'{self.left}, {self.right}'


def plot_regression_results(ax, y_true, y_pred, scores, name, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title = name + '\n Evaluation in {:.2f} seconds'.format(elapsed_time)
    ax.set_title(title)


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
