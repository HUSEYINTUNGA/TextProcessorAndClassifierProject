import os
import pandas as pd
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .general import detect_language, clean_text
import logging
from train_models import predict_class

logging.basicConfig(
    filename='error_logs.log', 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

user_csv_file = None
selected_columns = None

def classify_csv_upload(request):
    """
    Kullanıcıdan CSV dosyasını yüklemesini isteyen ve bu dosyanın sütunlarını döndüren fonksiyon.

    İşlevi:
    - Kullanıcının yüklediği CSV dosyasının formatını doğrular.
    - Dosyadaki sütunları analiz ederek kullanıcıya seçim için sunar.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Sütun bilgilerini içeren sınıflandırma sayfası veya hata mesajı.
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
                return render(request, 'classify_csv.html', {'columns': columns, 'message': 'CSV dosyası başarıyla yüklendi.'})
            except Exception as e:
                logging.error(f"CSV dosyası işlenirken hata: {e}")
                return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=400)

        return render(request, 'classify_csv.html')
    except Exception as e:
        logging.error(f"classify_csv_page fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})

def classify_select_columns(request):
    """
    Kullanıcının seçtiği sütunları işleyerek global değişkene kaydeden fonksiyon.

    İşlevi:
    - Kullanıcının seçtiği sütunların geçerliliğini kontrol eder.
    - Seçilen sütunları daha sonraki işlemler için global değişkene kaydeder.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Seçim sayfası veya hata mesajı.
    """
    global user_csv_file, selected_columns
    try:
        if request.method == 'POST':
            selected_columns = request.POST.getlist('selected_columns')

            if not selected_columns:
                columns = user_csv_file.columns.tolist()
                return render(request, 'classify_csv.html', {
                    'columns': columns,
                    'error_message': 'Hiçbir sütun seçilmedi. Lütfen en az bir sütun seçin.'
                })

            return render(request, 'classify_csv.html', {
                'selected_columns': selected_columns,
                'columns': user_csv_file.columns.tolist()
            })

        return render(request, 'classify_csv.html')
    except Exception as e:
        logging.error(f"select_columns fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})

def classify_csv_data(request):
    """
    Kullanıcının seçtiği sütunlarda sınıf tahmini yapar ve işlenmiş CSV dosyasını döndüren fonksiyon.

    İşlevi:
    - Orijinal metin verilerini işlenmek üzere hazırlar.
    - Metinleri temizlemek ve işlemek için `clean_text` fonksiyonunu kullanır.
    - İşlenmiş verilerle dil tespiti ve sınıf tahmini yapar.
    - Tahmin yapılamayan verileri işaretler.
    - İşlenmiş ve sınıflandırılmış CSV dosyasını kullanıcıya sunar.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: İşlenmiş CSV dosyasını veya hata mesajı.
    """
    global user_csv_file, selected_columns

    try:
        if user_csv_file is None:
            return JsonResponse({'error': 'CSV dosyası yüklenmemiş.'}, status=400)

        if not selected_columns:
            return JsonResponse({'error': 'Sütun seçilmemiş. Lütfen önce sütun seçimi yapın.'}, status=400)

        for column in selected_columns:
            if column not in user_csv_file.columns:
                return JsonResponse({'error': f"'{column}' sütunu bulunamadı."}, status=400)

            processed_texts = []
            predicted_classes = []

            for row in user_csv_file[column]:
                original_text = str(row)

                detected_language = detect_language(original_text)
                if detected_language not in ['en', 'tr']:
                    processed_texts.append("Dil desteklenmiyor")
                    predicted_classes.append("Tahmin yapılmadı")
                    continue

                processed_text = clean_text(original_text, {
                    'remove_punctuation': True,
                    'remove_special_chars': True,
                    'convert_to_lowercase': True,
                    'remove_stopwords': True,
                    'stemming': False,
                    'lemmatization': True
                }, detected_language)

                try:
                    predicted_class = predict_class(processed_text, detected_language)
                except Exception as e:
                    logging.error(f"Tahmin sırasında hata: {e}")
                    processed_texts.append(processed_text)
                    predicted_classes.append("Tahmin yapılamadı")
                    continue

                processed_texts.append(processed_text)
                predicted_classes.append(predicted_class)

            user_csv_file[f"{column}_processed_text"] = processed_texts
            user_csv_file[f"{column}_predicted_class"] = predicted_classes

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_filename = f"classifiedCsv_{timestamp}.csv"
        output_path = os.path.join('UserCsvFiles/ClassifyFiles', output_filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        user_csv_file.to_csv(output_path, index=False)

        with open(output_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{output_filename}"'
            return response

    except Exception as e:
        logging.error(f"classify_columns fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'}, status=500)
