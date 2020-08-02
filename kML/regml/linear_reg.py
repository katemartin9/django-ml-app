from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from regml.models import RegData, FileMetaData
import pandas as pd

# STEP 1: Retrieve data from db
# TODO:cache title
title = 'housing project'

data_dict = {}
for i, row in enumerate(RegData.objects.raw('SELECT * FROM regml_regdata WHERE project_name_id = %s', [title])):
    row_x = row.x
    row_x.append(row.y)
    data_dict[i] = row_x

for name in FileMetaData.objects.raw('SELECT * FROM regml_filemetadata WHERE project_name = %s', [title]):
    x_cols = name.col_names
    y_name = name.y_name
    x_cols.append(y_name)

df_original = pd.DataFrame(data_dict.values(), columns=x_cols)

# STEP 2 build data frame with features for training
# TODO: return column names and get user to choose from a drop down (numeric, categoric, datetime, specify format)
cols = {'numeric': ['rooms', 'distance', 'bedroom2', 'bathroom', 'car', 'landsize',
                      'buildingarea', 'lattitude', 'longtitude',
                      'propertycount', 'price'],
        'categorical': ['suburb', 'address', 'type', 'method', 'sellerg',
                        'postcode', 'regionname', 'councilarea'],
        'dates': ['date', 'yearbuilt']}

df = df_original.copy()
categorical_features = []
numeric_features = []
columns_to_drop = []

# numerical
for col in cols['numeric']:
    df[col] = pd.to_numeric(df[col], errors='ignore')
    if col != y_name:
        numeric_features.append(col)
# df_numeric = df_original.copy()[cols['numeric']]

# categorical
for col in cols['categorical']:
    vals = df[col].nunique()
    if vals < 10:
        categorical_features.append(col)
    else:
        columns_to_drop.append(col)
#df = pd.get_dummies(df, columns=categorical_features)

# dates
columns_to_drop.extend(cols['dates'])

# drop columns
df.drop(columns_to_drop, axis=1, inplace=True)

# TODO: build corr matrix showing correlated features

# STEP 3: Train test split
X, y = df.drop(y_name, axis=1), df[y_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# STEP 4: BUILD PIPELINE
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

regressor = Pipeline([('preprocessor', preprocessor),
                 ('regressor', LinearRegression())])

regressor.fit(X_train, y_train)
print("model score: %.3f" % regressor.score(X_test, y_test))