from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm, ColumnTypesForm
from django.contrib.auth.models import User
from .utils import handle_uploaded_file, db_load_file, db_load_column_types
from .tables import generate_table
from .models import RegData
import dateparser
import datetime


# Create your views here.
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            title = form.cleaned_data.get('title')
            tick = form.cleaned_data.get('tick')
            data = handle_uploaded_file(request.FILES['file'], tick)
            # TODO: check why doesn't throw error when project name exists
            user = User.objects.get()
            db_load_file(data, title, user)
            return HttpResponseRedirect(f'/data-preview/{title}')
        else:
            print('Error on form: ', form.errors)
            warning_message = f"Not a valid file."
            form = UploadFileForm()
            return render(request, 'upload_file.html', {'form': form, 'warning': warning_message})
    else:
        form = UploadFileForm()
    return render(request, 'upload_file.html', {'form': form})


def show_table(request, title):
    data = RegData.objects.all().filter(project_name=title).only('observations')
    results = []
    for d in data:
        results.append(d.observations)
    data_table = generate_table(results)
    forms = generate_column_types(request, results[0].keys(), results[0].values())
    if request.method == 'POST':
        data = []
        for i in range(len(forms)):
            form = ColumnTypesForm(request.POST, prefix=f'form_{i}')
            if form.is_valid():
                data.append(form.cleaned_data)
            else:
                print('Error on form: ', form.errors)
        db_load_column_types(data, title)
    return render(request, 'results.html', {'table': data_table, 'forms': forms})


def generate_column_types(request, columns, vals):
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

