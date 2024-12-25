import re
import string
from django.shortcuts import render
from django.http import JsonResponse
from nltk.corpus import stopwords
from googletrans import Translator
from nltk.stem import PorterStemmer, WordNetLemmatizer
from train_models import predict_class

translator = Translator()

def is_valid_word(word):
    """
    Kelimenin geçerli bir kelime olup olmadığını kontrol eder.
    Geçerli bir kelime, sadece harflerden oluşmalı ve uzunluğu 1'den büyük olmalıdır.
    """
    return len(word) > 1 and word.isalpha()

def detect_language(user_text):
    """
    Kullanıcının metninin dilini tespit eder.
    """
    try:
        detected_lang = translator.detect(user_text).lang
        print("Tespit edilen dil: ", detected_lang)
        return detected_lang
    except Exception as e:
        return f"Language detection error: {str(e)}"

def clean_text(user_text, options, language='en'):
    """
    Metni temizler ve dil desteği eklenir.
    """
    try:
        if options.get('remove_punctuation'):
            user_text = user_text.translate(str.maketrans('', '', string.punctuation))

        if options.get('remove_special_chars'):
            user_text = re.sub(r'[#$@{}\[\]/\\)(<>|!\'^+%&/\u00bd=*&\u20ac~\u00a8\u00b4\u00e6\u00a3\u00e9\u00df]', '', user_text)

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
            'classify_text': request.POST.get('classify_text') == 'on',
        }

        if not any(options.values()) and not request.POST.get('classify_text'):
            return JsonResponse({'error': 'Lütfen en az bir işlem türü veya sınıflandırma seçin.'})
        
        text_language = detect_language(user_text)
        processed_text = clean_text(user_text, options, text_language)
        classification_result = None

        if request.POST.get('classify_text'):
            try:
                classification_result = predict_class(user_text, text_language)
            except Exception as e:
                return JsonResponse({'error': f"Sınıf tahmini sırasında bir hata oluştu: {str(e)}"})

        return JsonResponse({
            'processed_text': processed_text,
            'classification_result': classification_result
        })

    return render(request, 'Home.html')

def AboutMe(request):
    """
    Hakkımda sayfasını render eder.
    Bu sayfa, uygulama ve geliştirici hakkında bilgi içermektedir.
    """
    return render(request, 'AboutMe.html')