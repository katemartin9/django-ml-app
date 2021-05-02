import nbformat as nbf
import zipfile
import io


def create_csv(data):
    f = io.StringIO()
    data.to_csv(f)
    return f


def _build_notebook(*args):
    nb = nbf.read('regml/template.ipynb', as_version=4)
    nb['cells'][4]['source'] = f"filename = 'dataset.csv'\ny = {args[0]}"
    return nbf.write(nb, 'sample_notebook.ipynb')


def _download_data(data, y):
    req_f = io.StringIO()
    req_f.write("scikit-learn==0.23.1\nplotly==4.14.3\njupyter==1.0.0")
    req_f.name = 'requirements.txt'
    files = ['regml/sample_notebook.ipynb', 'regml/utils.py', 'regml/notebook_helper.py']
    _build_notebook(y)
    compression = zipfile.ZIP_DEFLATED
    zf = zipfile.ZipFile("regml/static/regml/output.zip", mode="w")
    for f in files:
        zf.write(f, compress_type=compression)
    zf.writestr(req_f.name, req_f.getvalue(), compress_type=compression)
    zf.writestr('dataset.csv', create_csv(data).getvalue(), compress_type=compression)
    zf.close()