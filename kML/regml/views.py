from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
from .utils import *


# Create your views here.
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            y_field = form.cleaned_data.get('y')
            title = form.cleaned_data.get('title')
            tick = form.cleaned_data.get('tick')
            d_x, d_y = handle_uploaded_file(request.FILES['file'], y_field, tick)
            # TODO: check why doesn't throw error when project name exists
            load_file_metadata(d_x['columns'], y_field, title)
            load_file_into_db(d_x['data'], d_y, title)
            return HttpResponseRedirect('/results/')
        else:
            print('Error on form: ', form.errors)
            warning_message = f"Not a valid file."
            form = UploadFileForm()
            return render(request, 'upload_file.html', {'form': form, 'warning': warning_message})
    else:
        form = UploadFileForm()
    return render(request, 'upload_file.html', {'form': form})
