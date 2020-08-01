from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from .models import RegData, FileMetaData
import pandas as pd

# TODO:cache title
title = 'housing project'

data_dict = {}
for i, row in enumerate(RegData.objects.raw('SELECT * FROM regml_regdata'
                                            'WHERE project_title = %s', [title])):
    row_x = row.x
    row_x.append(row.y)
    data_dict[i] = row_x

for name in FileMetaData.objects.raw('SELECT * FROM regml_filemetadata'
                                     'WHERE project_title = %s', [title]):
    x_cols = name.col_names
    x_cols.append(name.y_name)

df = pd.DataFrame(data_dict.values(), columns=x_cols)