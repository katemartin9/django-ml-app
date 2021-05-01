import nbformat as nbf
import zipfile
import io


def build_notebook(filename, y):
    nb = nbf.read('template.ipynb', as_version=4)
    nb['cells'][0]['source'] = 'Replaced text'
    return nbf.write(nb, 'sample_notebook.ipynb')


def _download_data(*args):
    req_f = io.StringIO()
    req_f.write("scikit-learn==0.23.1\nplotly==4.14.3\njupyter==1.0.0")
    req_f.name = 'requirements.txt'
    files = ['sample_notebook.ipynb', 'utils.py', 'notebook_helper.py']
    build_notebook(args[0], args[1])
    compression = zipfile.ZIP_DEFLATED
    zf = zipfile.ZipFile("output.zip", mode="w")
    for f in files:
        zf.write(f, compress_type=compression)
    zf.writestr(req_f.name, req_f.getvalue(), compress_type=compression)
    zf.close()


if __name__ == '__main__':
    _download_data('datasets/boston_housing.csv', 'charges')