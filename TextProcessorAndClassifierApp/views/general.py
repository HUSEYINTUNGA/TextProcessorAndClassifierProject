import logging
from django.shortcuts import render
from django.http import JsonResponse
from TextProcessorAndClassifierApp.base_options import clean_text, detect_language, predict_class

logging.basicConfig(
    filename='error_logs.log', 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
                    if text_language=='en':
                        path_model='english_model.joblib'
                    else:
                        path_model='turkish_model.joblib' 
                    classification_result = predict_class(user_text,path_model,text_language)       
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

