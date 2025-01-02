from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from TextProcessorAndClassifierApp.base_options import load_data_from_db
import asyncio
import logging
import pandas as pd

user_csv_file = None
app_dataset = None
target_column = None

def select_using_dataset(request):
    """
    Kullanıcıya uygulama veri setini mi yoksa kendi veri setini mi kullanmak istediğini seçtiren fonksiyon.

    İşlevi:
    - Kullanıcının seçimine göre uygulama veri setini yükler veya kullanıcıya kendi veri setini yükleme seçeneği sunar.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Seçim sayfası veya hata mesajı.
    """
    global app_dataset
    if request.method == 'POST':
        dataset_option = request.POST.get('dataset_option', None)
        if dataset_option == 'app_dataset':
            language = request.POST.get('language', None)
            if language:
                try:
                    app_dataset = asyncio.run(load_data_from_db(language=language))
                    return render(request, 'choiceTrainDataset.html', {'dataset_loaded': True, 'use_custom_dataset': False})
                except Exception as e:
                    return HttpResponse(f"Dataset yüklenirken hata oluştu: {e}", status=500)
            else:
                return JsonResponse({'error': 'Lütfen bir CSV dosyası yükleyin.'}, status=400)
        elif dataset_option == 'custom_dataset':
            print("Kullanıcı kendi datasetini yüklemek istedi")
            return render(request, 'choiceTrainDataset.html', {'use_custom_dataset': True})
        else:
            return HttpResponse("Geçerli bir seçim yapılmadı.", status=400)
    return render(request, 'choiceTrainDataset.html')

def custom_upload_dataset(request):
    """
    Kullanıcının kendi CSV dosyasını yüklemesini sağlayan fonksiyon.

    İşlevi:
    - Kullanıcının yüklediği CSV dosyasının formatını kontrol eder.
    - Dosyayı okuyarak kullanılabilir hale getirir.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Sütun bilgilerini içeren seçim sayfası veya hata mesajı.
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
                return render(request, 'choiceTrainDataset.html', {
                    'use_custom_dataset': False,
                    'columns': columns,
                    'message': 'CSV dosyası başarıyla yüklendi.'
                })
            except Exception as e:
                logging.error(f"CSV dosyası işlenirken hata: {e}")
                return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=400)

        return render(request, 'choiceTrainDataset.html', {'use_custom_dataset': True})
    except Exception as e:
        logging.error(f"custom_dataset fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})

def select_target_column(request):
    """
    Kullanıcının veri setinden hedef sütunu seçmesini sağlayan fonksiyon.

    İşlevi:
    - Kullanıcının seçtiği sütunun doğruluğunu kontrol eder.
    - Hedef sütunu belirleyerek sınıf dağılımını hesaplar.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Hedef sütun ve sınıf dağılımını içeren seçim sayfası veya hata mesajı.
    """
    global user_csv_file, target_column
    try:
        if request.method == 'POST':
            choise_target_column = request.POST.get('selected_column', None)
            print("Seçilen sütun", choise_target_column)
            if not choise_target_column:
                columns = user_csv_file.columns.tolist()
                return render(request, 'choiceTrainDataset.html', {
                    'columns': columns,
                    'error_message': 'Hiçbir sütun seçilmedi. Lütfen en az bir sütun seçin.'
                })
            target_column = choise_target_column
            class_distribution = user_csv_file[target_column].value_counts().to_dict()
            return render(request, 'choiceTrainDataset.html', {
                'selected_column': target_column,
                'columns': None,
                'balance_step': True,
                'class_distribution': class_distribution,
            })
        return render(request, 'choiceTrainDataset.html')
    except Exception as e:
        logging.error(f"select_columns fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})

def balance_dataset(request):
    """
    Kullanıcının veri setindeki sınıfları dengelemesini sağlayan fonksiyon.

    İşlevi:
    - Veri setindeki sınıfların dağılımını kontrol eder.
    - Kullanıcının seçimine göre sınıf dağılımını dengeler veya olduğu gibi bırakır.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Dengelenmiş veri seti bilgilerini içeren sayfa veya hata mesajı.
    """
    global user_csv_file, target_column
    try:
        if request.method == 'POST':
            action = request.POST.get('action', None)
            print("Seçilen dengeleme adımı: ", action)
            if not action:
                return JsonResponse({'error': 'Bir seçenek seçmelisiniz.'}, status=400)

            target_column = target_column
            class_distribution = user_csv_file[target_column].value_counts().to_dict()

            if action == 'remove_low_classes':
                average_count = sum(class_distribution.values()) / len(class_distribution)
                user_csv_file = user_csv_file[user_csv_file[target_column].map(
                    lambda x: class_distribution.get(x, 0) >= average_count
                )]
                message = "Ortalamanın altında kalan sınıflar kaldırıldı."
            elif action == 'balance_distribution':
                min_count = min(class_distribution.values())
                user_csv_file = pd.concat([
                    user_csv_file[user_csv_file[target_column] == cls].sample(n=min_count, random_state=42)
                    for cls in class_distribution.keys()
                ])
                message = "Veri dağılımı dengelendi."
            elif action == 'use_as_is':
                user_csv_file=user_csv_file
                message = "Veri seti olduğu kullanıldı"
            else:
                return JsonResponse({'error': 'Geçersiz seçenek.'}, status=400)

            updated_class_distribution = user_csv_file[target_column].value_counts().to_dict()
            print("User csv file global değişkeninin içeriği \n", user_csv_file)
            return render(request, 'choiceTrainDataset.html', {
                'updated_class_distribution': updated_class_distribution,
                'message': message,
                'columns': None,
                'selected_column': None,
                'class_distribution': None,
            })
        else:
            return JsonResponse({'error': 'Yalnızca POST istekleri destekleniyor.'}, status=405)
    except Exception as e:
        logging.error(f"balance_dataset fonksiyonunda hata: {e}")
        return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=500)
