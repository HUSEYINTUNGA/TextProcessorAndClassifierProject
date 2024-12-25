import os
import pandas as pd
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .general import detect_language, clean_text

user_csv_file = None
selected_columns = None
data_language = None

def upload_csv_page(request):
    """
    Kullanıcının bir CSV dosyası yüklemesine olanak tanır.
    Yüklenen CSV dosyasının sütun isimlerini alır ve ekranda gösterir.
    """
    print("upload_csv_page fonksiyonu çalıştı")
    global user_csv_file
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']

        if not csv_file.name.endswith('.csv'):
            return JsonResponse({'error': 'Lütfen bir CSV dosyası yükleyin.'}, status=400)

        try:
            user_csv_file = pd.read_csv(csv_file)
            columns = user_csv_file.columns.tolist()
            return render(request, 'upload_csv.html', {'columns': columns, 'message': 'CSV dosyası başarıyla yüklendi.'})
        except Exception as e:
            return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=400)
    return render(request, 'upload_csv.html')

def process_columns(request):
    """
    Kullanıcının yüklenen CSV dosyasından sütunları seçmesine olanak tanır.
    Seçilen sütunlar global bir değişkene kaydedilir.
    """
    global user_csv_file
    global selected_columns
    if request.method == 'POST':
        selected_columns = request.POST.getlist('selected_columns')

        if not selected_columns:
            return JsonResponse({'error': 'Hiçbir sütun seçilmedi.'}, status=400)
        for column in selected_columns:
            print("Seçilen sütun/sütunlar : {}".format(column))

        return render(request, 'upload_csv.html', {'selected_columns': selected_columns, 'columns': user_csv_file.columns.tolist()})
    return render(request, 'upload_csv.html')

def process_csv(request):
    """
    Seçilen sütunlardaki verileri temizler ve belirlenen işlemleri uygular.
    İşlenmiş veriler yeni bir CSV dosyasına kaydedilir ve kullanıcıya indirme bağlantısı sağlanır.
    """
    global selected_columns
    global user_csv_file

    if request.method == 'POST':
        if user_csv_file is None:
            return JsonResponse({'error': 'CSV dosyası yüklenmemiş.'}, status=400)
        print(selected_columns)    
        operations = {
            'remove_punctuation': request.POST.get('remove_punctuation') == 'on',
            'remove_special_chars': request.POST.get('remove_special_chars') == 'on',
            'convert_to_lowercase': request.POST.get('convert_to_lowercase') == 'on',
            'convert_to_uppercase': request.POST.get('convert_to_uppercase') == 'on',
            'remove_stopwords': request.POST.get('remove_stopwords') == 'on',
            'stemming': request.POST.get('stemming') == 'on',
            'lemmatization': request.POST.get('lemmatization') == 'on',
        }
        if not any(operations.values()):
            return JsonResponse({'error': 'Lütfen en az bir işlem seçin.'}, status=400)
            
        try:
            for column in selected_columns:
                if column not in user_csv_file.columns:
                    return JsonResponse({'error': f"'{column}' sütunu bulunamadı."}, status=400)
                for index in range(len(user_csv_file[column])):
                    original_value = user_csv_file[column].iloc[index]
                    processed_value = clean_text(original_value, operations)
                    user_csv_file.at[index, column] = processed_value
                    
                    
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_filename = f'processedCsv_{timestamp}.csv'
            output_path = os.path.join('MetinApp/UserCsvFiles', output_filename)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            user_csv_file.to_csv(output_path, index=False)

            with open(output_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="{output_filename}"'
                return response

        except Exception as e:
            return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=400)


    return JsonResponse({'error': 'Geçerli bir istek yapılmadı.'})