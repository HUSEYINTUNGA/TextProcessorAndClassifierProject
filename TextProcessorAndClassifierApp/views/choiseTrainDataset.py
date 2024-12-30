from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from TextProcessorAndClassifierApp.base_options import load_data_from_db
import asyncio
import logging
import pandas as pd

user_csv_file = None
app_dataset = None
selected__target_column = None

def select_dataset(request):
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
                return HttpResponse("Lütfen bir dil seçiniz.", status=400)
        elif dataset_option == 'custom_dataset':
            return render(request, 'choiceTrainDataset.html', {'use_custom_dataset': True})
        else:
            return HttpResponse("Geçerli bir seçim yapılmadı.", status=400)
    return render(request, 'choiceTrainDataset.html')


def custom_upload_dataset(request):
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
    global user_csv_file, selected_target_column
    try:
        if request.method == 'POST':
            selected_target_column = request.POST.getlist('selected_columns')

            if not selected_target_column:
                columns = user_csv_file.columns.tolist()
                return render(request, 'choiceTrainDataset.html', {
                    'columns': columns,
                    'error_message': 'Hiçbir sütun seçilmedi. Lütfen en az bir sütun seçin.'
                })
            target_column = selected_target_column[0]
            class_distribution = user_csv_file[target_column].value_counts().to_dict()
            return render(request, 'choiceTrainDataset.html', {
                'selected_columns': selected_target_column,
                'columns': None,
                'balance_step': True,
                'class_distribution': class_distribution,
            })

        return render(request, 'choiceTrainDataset.html')
    except Exception as e:
        logging.error(f"select_columns fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})
    
def balance_dataset(request):
    global user_csv_file, selected_target_column
    try:
        if request.method == 'POST':
            action = request.POST.get('action', None)
            if not action:
                return JsonResponse({'error': 'Bir seçenek seçmelisiniz.'}, status=400)

            target_column = selected_target_column[0]
            class_distribution = user_csv_file[target_column].value_counts().to_dict()

            if action == 'remove_low_classes':
                average_count = sum(class_distribution.values()) / len(class_distribution)
                user_csv_file = user_csv_file[user_csv_file[target_column].map(
                    lambda x: class_distribution.get(x, 0) >= average_count
                )]
                message = "Ortalamanın altında kalan sınıflar kaldırıldı."
            elif action == 'balance_distribution':
                # Sınıf sayısını en az veri içeren sınıfa eşitle
                min_count = min(class_distribution.values())
                user_csv_file = pd.concat([
                    user_csv_file[user_csv_file[target_column] == cls].sample(n=min_count, random_state=42)
                    for cls in class_distribution.keys()
                ])
                message = "Veri dağılımı dengelendi."
            elif action == 'use_as_is':
                message = "Veri olduğu gibi kullanıldı."
            else:
                return JsonResponse({'error': 'Geçersiz seçenek.'}, status=400)

            updated_class_distribution = user_csv_file[target_column].value_counts().to_dict()

            return render(request, 'choiceTrainDataset.html', {
                'updated_class_distribution': updated_class_distribution,
                'message': message,
                'columns': None,
                'selected_columns': None,
                'class_distribution': None,
            })
        else:
            return JsonResponse({'error': 'Yalnızca POST istekleri destekleniyor.'}, status=405)
    except Exception as e:
        logging.error(f"balance_dataset fonksiyonunda hata: {e}")
        return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=500)