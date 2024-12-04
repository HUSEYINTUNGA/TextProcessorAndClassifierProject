from django.shortcuts import render

def HomePage(request):
    return render(request, 'Home.html')

def UploadPage(request):
    return render(request, 'UploadCsv.html')

def process_columns(request):
    return render(request, 'ProcessColumns.html')

def process_text(request):
    return render (request, 'ProcessTexts.html')

def predict_class(request):
    return render(request, 'PredictClasses.html')

def AboutMePage(request):
    return render(request, 'AboutMe.html')