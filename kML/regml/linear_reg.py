from regml.models import RegData, FileMetaData
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
import numpy as np
from kML.regml.utils import Container, plot_regression_results
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error

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

# STEP 3: Add numerical columns and analyse
if len(cols['numeric']) > 0:
    for col in cols['numeric']:
        df[col] = pd.to_numeric(df[col])
        if col != y_name:
            numeric_features.append(col)


# correlation matrix
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True)
#plt.show()
corr_matrix = pd.melt(df.corr().reset_index(), id_vars='index')
corr_matrix = corr_matrix[corr_matrix.value < 1.0]
high_pos_corr = list(corr_matrix[corr_matrix.value > 0.8][['index', 'variable']]\
    .itertuples(index=False, name=None))
corr_with_y = corr_matrix[(corr_matrix.variable == y_name) &
                          ((corr_matrix.value > 0.4) | (corr_matrix.value < -0.4))]['index'].to_list()
features = [Container(x) for x in high_pos_corr]
#columns_to_drop.extend(highly_corr)
#columns_to_drop.extend(['TAX', 'RAD', 'AGE'])

# Visualize the relationship between x and y
temp_feats = numeric_features.copy()
temp_feats.append(y_name)
normalized_df = (df[temp_feats] -
                 df[temp_feats].mean())/df[temp_feats].std()
x_vars = []
for i, f in enumerate(normalized_df.columns):
    if i % 5 == 0 and i != 0:
        sns.pairplot(normalized_df, x_vars=x_vars,
                     y_vars=[y_name],
                     height=7, aspect=0.7)
        #plt.show()
        x_vars = []
    else:
        x_vars.append(f)


# STEP 4: add categorical columns
if len(cols['categorical']) > 0:
    for col in cols['categorical']:
        vals = df[col].nunique()
        if vals < 10:
            categorical_features.append(col)
        else:
            columns_to_drop.append(col)

# STEP 5: drop columns that are not relevant
# drop dates
if len(cols['dates']) > 0:
    columns_to_drop.extend(cols['dates'])

# drop selected columns
df.drop(columns_to_drop, axis=1, inplace=True)

# STEP 6: Train test split
X, y = df.drop(y_name, axis=1), df[y_name].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# STEP 7: BUILD PIPELINE
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = make_column_transformer(
        (numeric_transformer, list(set(numeric_features).intersection(df.columns))),
        (categorical_transformer, list(set(categorical_features).intersection(df.columns))),
    remainder='passthrough')

# STEP 8: RUN THROUGH ESTIMATORS
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

# STEP 9: PLOT THE RESULTS
fig, axs = plt.subplots(2, 2, figsize=(9, 7))
axs = np.ravel(axs)
for ax, estimator in zip(axs, [rf_pipeline, regressor, gradient_pipeline, poly_pipeline]):
    start_time = time.time()
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    name = estimator[-1].__class__.__name__
    train_score = estimator.score(X_train, y_train)  # r2
    test_score = estimator.score(X_test, y_test)  # r2
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    elapsed_time = time.time() - start_time
    plot_regression_results(ax, y_test, y_pred,
                            (r'$R^2 train={:.2f}' + '\n' + '$R^2 test={:.2f}' + '\n' + r'$RMSE={:.2f}')
                            .format(train_score, test_score, rmse), name, elapsed_time),
plt.suptitle('Single predictors versus stacked predictors', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# STEP 10: OUTPUT RESULT




