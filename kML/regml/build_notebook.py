import nbformat as nbf

nb = nbf.v4.new_notebook()
nb['cells'] = []

# TEXT
heading = """**REGML NOTEBOOK**"""
introduction = """This is an interactive environment for you to play with your data"""
feature_selection = """**FEATURE SELECTION**"""
feature_selection_description = """This is a short description of what feature selection is"""

# CODE
imports = """import pandas as pd"""
data_preview = """df = pd.read_csv('datasets/boston_housing.csv')
df.head()
"""
data_describe = """# Take a look at statistics of your dataset
df.describe()
"""
data_info = """# Take a look at the format of your data columns
df.info()
"""

structure = {heading: 'text', introduction: 'text',
             imports: 'code',
             data_describe: 'code',
             data_preview: 'code',
             data_info: 'code',
             feature_selection:'text', feature_selection_description: 'text'
             }

for name, t in structure.items():
    if t == 'text':
        nb['cells'].append(nbf.v4.new_markdown_cell(name))
    elif t == 'code':
        nb['cells'].append(nbf.v4.new_code_cell(name))

nbf.write(nb, 'test2.ipynb')