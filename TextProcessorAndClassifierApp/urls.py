from django.urls import path
from TextProcessorAndClassifierApp.views.general import HomePage , AboutMe
from TextProcessorAndClassifierApp.views.csv_processing import process_csv_upload, process_select_columns, process_csv_data
from TextProcessorAndClassifierApp.views.csv_classify import classify_csv_upload, classify_select_columns, classify_csv_data
from TextProcessorAndClassifierApp.views.model_training import create_model, train_user_model, handle_user_satisfaction
from TextProcessorAndClassifierApp.views.predictions import upload_predict_csv,predict_select_columns,predict_csv_data

urlpatterns = [
    path('', HomePage, name='home'),
    path('about/', AboutMe, name='about'),

    path('process/upload_csv_file', process_csv_upload, name='prcs_upload_csv'),
    path('process/process_csv_columns', process_select_columns, name='prcs_select_columns'),
    path('process/process_csv_data/', process_csv_data, name='prcs_process_csv'),

    path('classify/upload_csv_file', classify_csv_upload, name='clsf_upload_csv'),
    path('classify/select_columns', classify_select_columns, name='clsf_select_columns'),
    path('classify/classify_columns/', classify_csv_data, name='clsf_classify_csv'),

    path('model/create/', create_model, name='create_model'), 
    path('model/train/', train_user_model, name='train_model'),
    path('model/satisfaction/', handle_user_satisfaction, name='handle_user_satisfaction'),
    path('model/upload_predict_csv/',upload_predict_csv, name='upload_predict_csv'),
    path('model/predict_select_columns',predict_select_columns, name='predict_select_columns'),
    path('model/predict_csv_data/', predict_csv_data, name='predict_csv_data'), 

]
