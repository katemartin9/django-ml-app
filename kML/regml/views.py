from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
from .utils import handle_uploaded_file, load_file_into_db


# Create your views here.
def upload_file(request):
    if request.method == 'POST':
        print(request.FILES)
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            y_field = form.cleaned_data.get('y')
            title = form.cleaned_data.get('title')
            tick = form.cleaned_data.get('tick')
            d_x, d_y = handle_uploaded_file(request.FILES['file'], y_field, tick)
            load_file_into_db(d_x, d_y, title)
            return HttpResponseRedirect('/results/')
        else:
            print('Error on form: ', form.errors)
            warning_message = f"Not a valid file."
            form = UploadFileForm()
            return render(request, 'upload_file.html', {'form': form, 'warning': warning_message})
    else:
        form = UploadFileForm()
    return render(request, 'upload_file.html', {'form': form})
