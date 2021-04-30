import nbformat as nbf

# TODO: export as a zip file with requirements.txt, notebook, data_preprocessing, modelling, plotting, csv


def build_notebook(filename, y):
    nb = nbf.v4.new_notebook()
    nb['cells'] = []

    # TEXT
    heading = """**REGRESSION NOTEBOOK**"""
    introduction = """This is an interactive environment for you to explore your data."""
    data_processing = """**PREPROCESSING**"""
    feature_selection = """**FEATURE SELECTION**"""
    feature_selection_description = """Feature selection is the process of 
selecting a subset of the most relevant features which should lead to the best model results."""

    # CODE
    imports = """import pandas as pd
from utils import DataFrameImputer
from notebook_helper import FeatureSelectionCl
    """

    data_preview = f"""# Let's take a look at the data.
df = pd.read_csv({filename})
df.head()"""
    data_describe = """# The statistics of your dataset
df.describe()"""
    data_info = """# These are your data columns
df.info()
    """

    imputer = """# Take care of your missing values by calling the built-in imputer, you can specify a threshold.
df = DataFrameImputer().res(df).df"""

    structure = {heading: 'text',
                 introduction: 'text',
                 imports: 'code',
                 data_processing: 'text',
                 data_preview: 'code',
                 data_describe: 'code',
                 data_info: 'code',
                 imputer: 'code',

                 feature_selection: 'text', feature_selection_description: 'text'
                 }

    for name, t in structure.items():
        if t == 'text':
            nb['cells'].append(nbf.v4.new_markdown_cell(name))
        elif t == 'code':
            nb['cells'].append(nbf.v4.new_code_cell(name))

    nbf.write(nb, 'test2.ipynb')


if __name__ == '__main__':
    build_notebook('datasets/boston_housing.csv', 'charges')