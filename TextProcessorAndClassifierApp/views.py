from django.shortcuts import render
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from django.http import JsonResponse, HttpResponse
import pandas as pd
import os
from django.conf import settings
from datetime import datetime
import time
user_csv_file = None
selected_columns = None
def is_valid_word(word):
    """
    Kelimenin geçerli bir kelime olup olmadığını kontrol eder.
    Geçerli bir kelime, sadece harflerden oluşmalı ve uzunluğu 1'den büyük olmalıdır.
    """
    return len(word) > 1 and word.isalpha()

def clean_text(user_text, options):
    """
    Parametre olarak gelen metni temizler ve belirli işlemleri uygular.
    Uygulanabilecek işlemler:
    - Noktalama işaretlerini kaldırma.
    - Özel karakterleri kaldırma.
    - Harfleri büyük/küçük hale dönüştürme.
    - Durdurma kelimelerini (stopwords) kaldırma.
    - Stemming ve lemmatization işlemleri.
    """
    try:
        if options.get('remove_punctuation'):
            user_text = user_text.translate(str.maketrans('', '', string.punctuation))

        if options.get('remove_special_chars'):
            user_text = re.sub(r'[#$@{}\[\]\/\\)<>(|!\'^+%&/½=*&€~¨´æ£éß]', '', user_text)

        if options.get('convert_to_lowercase'):
            user_text = user_text.lower()

        if options.get('convert_to_uppercase'):
            user_text = user_text.upper()

        if options.get('remove_stopwords'):
            stop_words = set(stopwords.words('english'))
            user_text = ' '.join([word for word in user_text.split() if word.lower() not in stop_words])

        if options.get('stemming'):
            ps = PorterStemmer()
            user_text = ' '.join([ps.stem(word) for word in user_text.split() if is_valid_word(word)])

        if options.get('lemmatization'):
            lemmatizer = WordNetLemmatizer()
            user_text = ' '.join([lemmatizer.lemmatize(word) for word in user_text.split() if is_valid_word(word)])
        
        return user_text
    except Exception as e:
        return f"Text cleaning error: {str(e)}"

def HomePage(request):
    """
    Ana sayfa işlemlerini yöneten fonksiyon.
    Kullanıcıdan metin alır, belirli temizleme işlemleri ve sınıflandırma uygular.
    İşlenmiş metni ve sınıflandırma sonucunu JSON formatında döndürür.
    """
    if request.method == 'POST':
        user_text = request.POST.get('text', '')

        if not user_text.strip():
            return JsonResponse({'error': 'Lütfen metin girin.'})

        options = {
            'remove_punctuation': request.POST.get('remove_punctuation') == 'on',
            'remove_special_chars': request.POST.get('remove_special_chars') == 'on',
            'convert_to_lowercase': request.POST.get('convert_to_lowercase') == 'on',
            'convert_to_uppercase': request.POST.get('convert_to_uppercase') == 'on',
            'remove_stopwords': request.POST.get('remove_stopwords') == 'on',
            'stemming': request.POST.get('stemming') == 'on',
            'lemmatization': request.POST.get('lemmatization') == 'on',
        }

        if not any(options.values()) and not request.POST.get('classify_text'):
            return JsonResponse({'error': 'Lütfen en az bir işlem türü veya sınıflandırma seçin.'})

        processed_text = clean_text(user_text, options)
        classification_result = None

        if request.POST.get('classify_text'):
            try:
                classification_result = predict_class(user_text)
            except Exception as e:
                return JsonResponse({'error': f"Sınıf tahmini sırasında bir hata oluştu: {str(e)}"})

        return JsonResponse({
            'processed_text': processed_text,
            'classification_result': classification_result
        })

    return render(request, 'Home.html')




from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd

# Global değişkenler
user_csv_file = None
selected_columns = None

def UploadPage(request):
    """
    CSV dosyası yükleme ve sütunları frontend'e aktarma
    """
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']

        # Dosya formatını kontrol et
        if not csv_file.name.endswith('.csv'):
            return render(request, 'Upload.html', {'error_message': 'Lütfen bir CSV dosyası yükleyin.'})

        try:
            global user_csv_file
            user_csv_file = pd.read_csv(csv_file)  # CSV dosyasını oku
            columns = user_csv_file.columns.tolist()  # Sütunları al

            if not columns:
                return render(request, 'Upload.html', {'error_message': 'Yüklenen CSV dosyasında sütun bulunamadı.'})
            print("Yüklenen dosyanın sütunları :" ,columns)
            # Sütunları template'e aktar
            return render(request, 'Upload.html', {
                'columns': columns,
                'message': 'CSV dosyası başarıyla yüklendi.'
            })
        except Exception as e:
            return render(request, 'Upload.html', {'error_message': f'Hata oluştu: {str(e)}'})

    return render(request, 'Upload.html')

def process_columns(request):
    global selected_columns
    if request.method == 'POST':
        selected_columns = request.POST.getlist('selected_columns')
        if not selected_columns:
            return render(request, 'Upload.html', {'error_message': 'Lütfen en az bir sütun seçin.'})
        print(selected_columns)
        return render(request, 'Upload.html', {
            'selected_columns': selected_columns,
            'message': 'Sütunlar başarıyla seçildi.',
            'show_operation_form': True 
        })

    return render(request, 'Upload.html')


def process_text(request):
    columns = request.GET.get('columns', [])
    # İşlem için gerekli adımları burada gerçekleştirebilirsiniz
    return render(request, 'process_text.html', {'columns': columns})

def predict_class(request):
    columns = request.GET.get('columns', [])
    # Sınıflandırma için gerekli adımları burada gerçekleştirebilirsiniz
    return render(request, 'PredictClasses.html', {'columns': columns})


def AboutMePage(request):
    return render(request, 'AboutMe.html')