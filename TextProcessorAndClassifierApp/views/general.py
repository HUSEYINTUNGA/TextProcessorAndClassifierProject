import re
import string
import logging
from nltk.corpus import stopwords
from googletrans import Translator
from django.shortcuts import render
from django.http import JsonResponse
from train_models import predict_class
from nltk.stem import PorterStemmer, WordNetLemmatizer

logging.basicConfig(
    filename='error_logs.log', 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

translator = Translator()

def is_valid_word(word):
    """
    Kelimenin geçerli bir kelime olup olmadığını kontrol eder.

    Geçerli bir kelime, yalnızca harflerden oluşan ve uzunluğu 1'den büyük olan bir kelimedir.
    Bu fonksiyon, kelimenin uygun bir formatta olup olmadığını doğrulamak için kullanılır.

    Parametreler:
        word (str): Doğrulanacak kelime.

    Döndürür:
        bool: Eğer kelime geçerliyse True, aksi halde False.
    """
    return len(word) > 1 and word.isalpha()

def detect_language(user_text):
    """
    Kullanıcının metninin dilini tespit eder.

    Google Translate API kullanarak metnin dilini algılar.
    Eğer hata oluşursa bunu loglar ve uygun bir hata mesajı döndürür.

    Parametreler:
        user_text (str): Dil tespiti yapılacak metin.

    Döndürür:
        str: Tespit edilen dilin kodu veya hata mesajı.
    """
    try:
        detected_lang = translator.detect(user_text).lang
        print("Tespit edilen dil: ", detected_lang)
        return detected_lang
    except Exception as e:
        logging.error(f"Dil tespiti sırasında hata: {e}")
        return "Language detection error"

def clean_text(user_text, options, language='en'):
    """
    Metni temizler ve belirtilen işlemleri uygular.

    Metin üzerinde noktalama işaretlerini kaldırma, büyük/küçük harf dönüşümü,
    durdurma kelimelerini çıkarma gibi işlemler gerçekleştirir.
    Ayrıca dil bilgisine uygun olarak kökleme (stemming) ve lemmatizasyon işlemleri uygulanabilir.

    Parametreler:
        user_text (str): İşlenecek metin.
        options (dict): Uygulanacak işlemleri belirten seçenekler.
        language (str): Metnin dili.

    Döndürür:
        str: İşlenmiş metin veya hata mesajı.
    """
    try:
        if options.get('remove_punctuation'):
            user_text = user_text.translate(str.maketrans('', '', string.punctuation))

        if options.get('remove_special_chars'):
            user_text = re.sub(r'[#$@{}\[\]/\\)(<>|!\'^+%&/½=*&\u20ac~\u00a8\u00b4\u00e6\u00a3\u00e9\u00df]', '', user_text)

        if options.get('convert_to_lowercase'):
            user_text = user_text.lower()

        if options.get('convert_to_uppercase'):
            user_text = user_text.upper()

        if options.get('remove_stopwords'):
            stop_words = set(stopwords.words(language)) if language in stopwords.fileids() else set()
            user_text = ' '.join([word for word in user_text.split() if word.lower() not in stop_words])

        if options.get('stemming'):
            ps = PorterStemmer()
            user_text = ' '.join([ps.stem(word) for word in user_text.split() if is_valid_word(word)])

        if options.get('lemmatization'):
            lemmatizer = WordNetLemmatizer()
            user_text = ' '.join([lemmatizer.lemmatize(word) for word in user_text.split() if is_valid_word(word)])

        return user_text
    except Exception as e:
        logging.error(f"Metin temizleme sırasında hata: {e}")
        return "Text cleaning error"

def HomePage(request):
    """
    Ana sayfa işlemlerini yöneten fonksiyon.

    Kullanıcıdan alınan metin üzerinde seçilen işlemleri uygular ve sınıflandırma yapar.
    İşlenmiş metni ve sınıflandırma sonucunu JSON formatında döndürür.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        JsonResponse: İşlenmiş metin ve sınıflandırma sonucunu içeren yanıt.
    """
    try:
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
                'classify_text': request.POST.get('classify_text') == 'on',
            }

            if not any(options.values()):
                return JsonResponse({'error': 'Lütfen en az bir işlem türü veya sınıflandırma seçin.'})

            text_language = detect_language(user_text)

            if text_language == "Language detection error":
                return JsonResponse({'error': 'Dil tespiti sırasında bir hata oluştu.'})

            processed_text = clean_text(user_text, options, text_language)

            if processed_text == "Text cleaning error":
                return JsonResponse({'error': 'Metin temizleme sırasında bir hata oluştu.'})

            if options.get('classify_text'):
                if text_language not in ['en', 'tr']:
                    return JsonResponse({'error': f'Girdiğiniz metnin dili "{text_language}", modeller tarafından desteklenmiyor.'})

                try:
                    classification_result = predict_class(user_text, text_language)
                    return JsonResponse({
                        'processed_text': processed_text,
                        'classification_result': classification_result,
                        'language_detected': text_language
                    })
                except Exception as e:
                    logging.error(f"Sınıflandırma sırasında hata: {e}")
                    return JsonResponse({'error': 'Sınıf tahmini sırasında bir hata oluştu.'})

            return JsonResponse({
                'processed_text': processed_text,
                'language_detected': text_language
            })

        return render(request, 'Home.html')
    except Exception as e:
        logging.error(f"HomePage fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})

def AboutMe(request):
    """
    Hakkımda sayfasını render eder.

    Bu sayfa, uygulama ve geliştirici hakkında bilgi içermektedir.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Hakkımda sayfasını içeren yanıt.
    """
    try:
        return render(request, 'AboutMe.html')
    except Exception as e:
        logging.error(f"AboutMe fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Sayfa yüklenirken bir hata oluştu.'})
