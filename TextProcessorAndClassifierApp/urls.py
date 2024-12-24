from django.urls import path
from . import views

urlpatterns = [
    path('', views.HomePage, name='home'),
    path('about/', views.AboutMe, name='about'),
]
