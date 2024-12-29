import os
import pandas as pd
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from TextProcessorAndClassifierApp.base_options import detect_language, clean_text

import logging

logging.basicConfig(
    filename='error_logs.log', 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

user_csv_file = None
selected_columns = None

def process_csv_upload(request):
    """
    CSV dosyasını yüklemek için kullanıcıdan giriş alır.

    Kullanıcının yüklediği dosyanın geçerliliğini kontrol eder ve sütun bilgilerini döndürür.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Yükleme sayfasını veya hata durumunda JsonResponse.
    """
    global user_csv_file
    try:
        if request.method == 'POST' and request.FILES.get('csv_file'):
            csv_file = request.FILES['csv_file']

            if not csv_file.name.endswith('.csv'):
                return JsonResponse({'error': 'Lütfen bir CSV dosyası yükleyin.'}, status=400)

            try:
                user_csv_file = pd.read_csv(csv_file)
                columns = user_csv_file.columns.tolist()
                return render(request, 'process_csv.html', {'columns': columns, 'message': 'CSV dosyası başarıyla yüklendi.'})
            except Exception as e:
                logging.error(f"CSV dosyası işlenirken hata: {e}")
                return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=400)

        return render(request, 'process_csv.html')
    except Exception as e:
        logging.error(f"upload_csv_page fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})

def process_select_columns(request):
    """
    Kullanıcının seçtiği sütunları işler ve bunları global değişkene kaydeder.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Seçim sayfasını veya hata durumunda JsonResponse.
    """
    global user_csv_file, selected_columns
    try:
        if request.method == 'POST':
            selected_columns = request.POST.getlist('selected_columns')

            if not selected_columns:
                return JsonResponse({'error': 'Hiçbir sütun seçilmedi.'}, status=400)

            return render(request, 'process_csv.html', {'selected_columns': selected_columns, 'columns': user_csv_file.columns.tolist()})

        return render(request, 'process_csv.html')
    except Exception as e:
        logging.error(f"process_columns fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})

def process_csv_data(request):
    """
    Seçilen sütunlardaki verileri temizler ve belirlenen işlemleri uygular.

    Her veri için dil tespiti yapılır ve buna göre en uygun metin işleme işlemleri gerçekleştirilir.

    İşlenmiş verileri bir CSV dosyasına kaydeder ve kullanıcıya indirme bağlantısı sunar.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: İşlenmiş CSV dosyasını veya hata durumunda JsonResponse.
    """
    global user_csv_file, selected_columns

    try:
        if request.method == 'POST':
            if user_csv_file is None:
                return JsonResponse({'error': 'CSV dosyası yüklenmemiş.'}, status=400)

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

                    def process_row(row):
                        language = detect_language(str(row))
                        if language == "Language detection error":
                            logging.error(f"Dil tespiti sırasında hata oluştu: {row}")
                            return row
                        return clean_text(str(row), operations, language)

                    user_csv_file[column] = user_csv_file[column].apply(process_row)

                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                output_filename = f'processedCsv_{timestamp}.csv'
                output_path = os.path.join('UserFiles/ProcessedCsvFiles', output_filename)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                user_csv_file.to_csv(output_path, index=False)

                with open(output_path, 'rb') as f:
                    response = HttpResponse(f.read(), content_type='text/csv')
                    response['Content-Disposition'] = f'attachment; filename="{output_filename}"'
                    return response

            except Exception as e:
                logging.error(f"CSV işlemleri sırasında hata: {e}")
                return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=400)

        return JsonResponse({'error': 'Geçerli bir istek yapılmadı.'})
    except Exception as e:
        logging.error(f"process_csv fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})
