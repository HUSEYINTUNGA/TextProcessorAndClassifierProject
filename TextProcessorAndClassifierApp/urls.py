from django.urls import path
from . import views

urlpatterns = [
    path('', views.HomePage, name='home'),
    path('about/', views.AboutMePage, name='about'),
    path('data/upload', views.UploadPage, name='upload'), 
    path('data/select-columns/', views.process_columns, name='process_columns'), 
    path('data/process-text/', views.process_text, name='process_text'),
    path('data/predict-class', views.predict_class, name='predict_class')
]