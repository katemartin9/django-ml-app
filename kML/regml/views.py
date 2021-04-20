from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from .forms import UploadFileForm, ColumnTypesForm, ColumnsToRemove
from django.contrib.auth.models import User
from .utils import handle_uploaded_file, db_load_file, db_load_column_types
from .tables import generate_table
from .models import RegData
import dateparser
import datetime
from .ml_models import FeatureSelection, RegModel
import pandas as pd
import json


# Create your views here.
# UPLOAD FILE VIEW - upload_file.html
def upload_file(request):
    """
    Step1: Renders the file upload form
    Step2: loads the supplied file into the database
    """
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            title = form.cleaned_data.get('title')
            tick = form.cleaned_data.get('tick')
            data = handle_uploaded_file(request.FILES['file'], tick)
            # TODO: check why doesn't throw error when project name exists
            user = User.objects.get(pk=request.user.id)
            db_load_file(data, title, user)
            return HttpResponseRedirect(reverse('data_preview', args=([title])))
        else:
            return render(request, 'upload_file.html', {'form': form})
    else:
        form = UploadFileForm()
    return render(request, 'upload_file.html', {'form': form})


# RENDER TABLE VIEW - results.html
def show_table(request, title):
    """
    Step 1: Renders a preview table with a snapshot of the data supplied
    Step2: Loads updated column types into the database
    """
    data = RegData.objects.all().filter(project_name=title).only('observations')
    results = []
    for d in data:
        results.append(d.observations)
    data_table = generate_table(results)
    forms = generate_column_types(request, results[0].keys(), results[0].values())
    if request.method == 'POST':
        data = []
        for i in range(len(forms)):
            form = ColumnTypesForm(request.POST, prefix=f'form_{i}', fullpath=request.get_full_path())
            if form.is_valid():
                data.append(form.cleaned_data)
            else:
                print('Error on form: ', form.errors)
        db_load_column_types(data, title)
        return HttpResponseRedirect(reverse('render_graphs', args=([title])))
    return render(request, 'results.html', {'table': data_table, 'forms': forms})


def generate_column_types(request, columns, vals):
    """This function guesses column type and creates a form for each column name"""
    forms = []
    customlist = ["%d-%m-%Y", "%d/%m/%Y", "%Y", "%m-%d-%Y", "%m/%d/%Y"]
    idx = 0
    for col, val in zip(columns, vals):
        if isinstance(val, float) or isinstance(val, int):
            forms.append(ColumnTypesForm(initial={'col_name': col, 'col_type': 'n'}, prefix=f'form_{idx}'))
        elif isinstance(dateparser.parse(val,
                                         settings={'PARSERS': ['custom-formats']},
                                         date_formats=customlist), datetime.date):
            forms.append(ColumnTypesForm(initial={'col_name': col, 'col_type': 'd'}, prefix=f'form_{idx}'))
        elif isinstance(val, str):
            forms.append(ColumnTypesForm(initial={'col_name': col, 'col_type': 'c'}, prefix=f'form_{idx}'))
        idx += 1
    return forms


# RENDER FEATURE SELECTION render_graphs.html
def render_graphs(request, title):
    context = dict()
    cl = FeatureSelection(title)
    context['corr_plot'], context['xy_plot'], context['f_plot'], cols_to_remove = cl.run()
    context['dist_div'] = cl.plot_distributions()
    context['forms'] = []
    for idx, col in enumerate(cl.x_cols):
        if col in cols_to_remove:
            context['forms'].append(ColumnsToRemove(initial={'col_name': col,
                                                             'remove_add': True}, prefix=f'form_{idx}'))
        else:
            context['forms'].append(ColumnsToRemove(initial={'col_name': col,
                                                             'remove_add': False}, prefix=f'form_{idx}'))
    if request.method == 'POST':
        data = []
        for i in range(len(context['forms'])):
            form = ColumnsToRemove(request.POST, prefix=f'form_{i}')
            if form.is_valid():
                data.append(form.cleaned_data['col_name']) if form.cleaned_data['remove_add'] else None
            else:
                print('Error on form: ', form.errors)
        request.session['drop_cols'] = data
        request.session['data'] = cl.df.to_json()
        request.session['col_types'] = cl.col_types
        request.session['y'] = cl.y_cols
        return HttpResponseRedirect(reverse('train_models', args=([title])))
    return render(request, 'render_graphs.html', context)


def train_models(request, title):
    context = dict()
    cols_to_drop = request.session['drop_cols']
    data = json.loads(request.session['data'])
    df = pd.DataFrame.from_dict(data)
    col_types = request.session['col_types']
    y = request.session['y'][0]
    reg_cl = RegModel()
    reg_cl.split_train_test(df, y, cols_to_drop, col_types['n'], col_types['c'])
    reg_cl.run()
    context['model_plot'] = reg_cl.plot_model_performance()
    return render(request, 'train_models.html', context)

