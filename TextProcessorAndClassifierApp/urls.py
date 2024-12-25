from django.urls import path
from TextProcessorAndClassifierApp.views.general import HomePage , AboutMe
from TextProcessorAndClassifierApp.views.csv_processing import process_csv_upload, process_select_columns, process_csv_data
from TextProcessorAndClassifierApp.views.csv_classify import classify_csv_upload, classify_select_columns, classify_csv_data
urlpatterns = [
    path('', HomePage, name='home'),
    path('about/', AboutMe, name='about'),
    path('process/upload_csv_file', process_csv_upload, name='prcs_upload_csv'),
    path('process/process_csv_columns', process_select_columns, name='prcs_select_columns'),
    path('process/process_csv_data/', process_csv_data, name='prcs_process_csv'),
    path('classify/upload_csv_file', classify_csv_upload, name='clsf_upload_csv'),
    path('classify/select_columns', classify_select_columns, name='clsf_select_columns'),
    path('classify/classify_columns/', classify_csv_data, name='clsf_classify_csv'),
]
