import nbformat as nbf
import zipfile
import io
import os


def create_csv(data):
    f = io.StringIO()
    data.to_csv(f)
    return f


def build_relative_path(*files):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, *files)


def _build_notebook(target: str):
    nb = nbf.read(build_relative_path('template.ipynb'), as_version=4)
    nb['cells'][4]['source'] = f"filename = 'dataset.csv'\ny = {target}"
    return nbf.write(nb, build_relative_path('sample_notebook.ipynb'))


def _download_data(data, y):
    req_f = io.StringIO()
    req_f.write("scikit-learn==0.23.1\nplotly==4.14.3\njupyter==1.0.0")
    req_f.name = 'requirements.txt'
    files = ['sample_notebook.ipynb', 'utils.py', 'notebook_helper.py']
    _build_notebook(y)
    compression = zipfile.ZIP_DEFLATED
    zf = zipfile.ZipFile(build_relative_path("static", "regml", "output.zip"), mode="w")
    for f in files:
        zf.write(build_relative_path(f), compress_type=compression)
    zf.writestr(req_f.name, req_f.getvalue(), compress_type=compression)
    zf.writestr('dataset.csv', create_csv(data).getvalue(), compress_type=compression)
    zf.close()


if __name__ == '__main__':
    res = build_relative_path()
    print(res)