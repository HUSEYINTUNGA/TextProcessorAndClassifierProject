from django.urls import path
from TextProcessorAndClassifierApp.views.general import HomePage , AboutMe
from TextProcessorAndClassifierApp.views.csv_processing import process_csv_upload, process_select_columns, process_csv_data
from TextProcessorAndClassifierApp.views.csv_classify import classify_csv_upload, classify_select_columns, classify_csv_data
from TextProcessorAndClassifierApp.views.create_dataset_and_model import select_using_dataset, custom_upload_dataset, select_target_column, balance_dataset
from TextProcessorAndClassifierApp.views.create_dataset_and_model import choice_model_components , handle_user_satisfaction
from TextProcessorAndClassifierApp.views.create_dataset_and_model import upload_predict_csv_file,select_predict_columns,classifier_csv_file


urlpatterns = [
    path('', HomePage, name='home'),
    path('about/', AboutMe, name='about'),

    path('process/uploadCsvFile', process_csv_upload, name='prcs_upload_csv'),
    path('process/selectColumns', process_select_columns, name='prcs_select_columns'),
    path('process/processColumns', process_csv_data, name='prcs_process_csv'),

    path('classify/uploadCsvFile', classify_csv_upload, name='clsf_upload_csv'),
    path('classify/selectColumns', classify_select_columns, name='clsf_select_columns'),
    path('classify/classifyColumns', classify_csv_data, name='clsf_classify_csv'),

    path('createModel/', select_using_dataset, name='select_using_dataset'),
    path('createModel/CustomDataset/uploadTrainDataset', custom_upload_dataset, name='custom_dataset'),
    path('createModel/CustomDataset/selectTargetColumn', select_target_column, name='select_target_column'),
    path('createModel/CustomDataset/balanceTrainDataset', balance_dataset, name='balance_dataset'),
    path('createModel/choiceModelComponents', choice_model_components, name='choice_model_components'),
    path('createModel/modelMetrics', handle_user_satisfaction, name='handle_user_satisfaction'),
    path('createModel/uploadCsvFile', upload_predict_csv_file, name='upload_predict_csv'),
    path('createModel/selectColumns', select_predict_columns, name='select_predict_columns'),
    path('createModel/predictColumns', classifier_csv_file, name='classifier_csv_file')
]
