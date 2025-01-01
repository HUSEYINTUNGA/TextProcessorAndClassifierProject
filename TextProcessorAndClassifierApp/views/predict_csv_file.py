import os
import logging
import pandas as pd
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from TextProcessorAndClassifierApp.base_options import detect_language, clean_text, predict_class
from TextProcessorAndClassifierApp.views.create_model import global_model_paths

logging.basicConfig(
    filename='error_logs.log', 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

user_csv_file = None
selected_columns = None

def upload_predict_csv_file(request):
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


def select_predict_columns(request):
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


def classifier_csv_file(request):
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

            predicted_classes = []

            for row in user_csv_file[column]:
                original_text = str(row)

                detected_language = detect_language(original_text)
                if detected_language not in ['en', 'tr']:
                    predicted_classes.append("Dil desteklenmiyor")
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
                    print(detected_language)
                    if detect_language=='tr':
                        path_model="turkish_model.joblib"
                    else:
                        path_model="english_model.joblib"
                    print("model path : ",path_model)
                    predicted_class = predict_class(processed_text,global_model_paths,detected_language)
                except Exception as e:
                    logging.error(f"Tahmin sırasında hata: {e}")
                    predicted_classes.append("Tahmin yapılamadı")
                    continue

                predicted_classes.append(predicted_class)

            result_column_name = f"{column}_class"
            user_csv_file[result_column_name] = predicted_classes

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_filename = f"classifiedCsv_{timestamp}.csv"
        output_path = os.path.join('UserFiles/ClassifyCsvFiles', output_filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        user_csv_file.to_csv(output_path, index=False)

        with open(output_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{output_filename}"'
            return response

    except Exception as e:
        logging.error(f"classify_csv_data fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'}, status=500)

    