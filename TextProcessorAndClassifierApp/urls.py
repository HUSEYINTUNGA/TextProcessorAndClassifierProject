from django.urls import path
from TextProcessorAndClassifierApp.views.general import HomePage , AboutMe
from TextProcessorAndClassifierApp.views.csv_processing import upload_csv_page, process_csv, process_columns

urlpatterns = [
    path('', HomePage, name='home'),
    path('about/', AboutMe, name='about'),
    path('csv/upload_csv_file', upload_csv_page, name='upload_csv'),
    path('csv/process_csv_columns', process_columns, name='process_columns'),
    path('csv/process_csv_data/', process_csv, name='process_csv'),
]
