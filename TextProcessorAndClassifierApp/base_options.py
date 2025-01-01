import os
import re
import string
import logging
import pandas as pd
from joblib import load
from nltk.corpus import stopwords
from googletrans import Translator
from asgiref.sync import sync_to_async
from nltk.stem import PorterStemmer, WordNetLemmatizer
from .models import UploadedEnglishTexts, UploadedTurkishTexts

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
            if language == 'tr':
                stop_words = set(stopwords.words('turkish'))
            elif language == 'en':
                stop_words = set(stopwords.words('english'))
            else:
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

convertEnglish = {
    "World": 1,
    "Sports": 2,
    "Business": 3,
    "Sci/Tech": 4,
    "Entertainment": 5,
    "Politics": 6,
}

convertTurkish = {
    "Ekonomi": 0,
    "Kültür-Sanat": 1,
    "Sağlık": 2,
    "Siyaset": 3,
    "Spor": 4,
    "Teknoloji": 5,
}

def predict_class(text, model_path, language='en'):
    """
    Girdi metnini sınıflandırır ve tahmin edilen sınıfı döndürür.

    Parametreler:
        text (str): Sınıflandırılacak metin.
        model_path (str): Kullanılacak modelin dosya yolu.
        language (str): 'en' veya 'tr' olarak dil seçimi.

    Döndürür:
        str: Tahmin edilen sınıf ismi.
    """
    if language == 'en' and model_path == 'english_model.joblib':
        model_path = os.path.join("ML_Model", "english_model.joblib")
        vectorizer_path = os.path.join("ML_Model", "english_vectorizer.joblib")
        class_map = {v: k for k, v in convertEnglish.items()}
    elif language == 'tr' and model_path == 'turkish_model.joblib':
        model_path = os.path.join("ML_Model", "turkish_model.joblib")
        vectorizer_path = os.path.join("ML_Model", "turkish_vectorizer.joblib")
        class_map = {v: k for k, v in convertTurkish.items()}
    else:
        model_path = os.path.join("UserFiles/UserModels", 'user_models.joblib')
        vectorizer_path = os.path.join("UserFiles/UserModels", 'user_vectorizer.joblib')
        class_map = {v: k for k, v in convertEnglish.items()}

    try:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = load(model_path)
            vectorizer = load(vectorizer_path)
        else:
            raise FileNotFoundError("Model veya vektörizer bulunamadı.")

        options = {
            'remove_punctuation': True,
            'remove_special_chars': True,
            'convert_to_lowercase': True,
            'remove_stopwords': True,
            'stemming': False,
            'lemmatization': True,
            }

        cleaned_text = clean_text(text, options, language)
        text_vector = vectorizer.transform([cleaned_text])
        predicted_class_int = model.predict(text_vector)[0]

        predicted_class_name = class_map.get(predicted_class_int, "Bilinmeyen Sınıf")
        return predicted_class_name
    except Exception as e:
        logging.error(f"Tahmin hatası: {e}")
        return "Tahmin Hatası"

async def load_data_from_db(language='en'):
    """
    Veritabanından dil bilgisine göre verileri yükler.

    Parametreler:
        language (str): 'en' veya 'tr' olarak dil seçimi.

    Döndürür:
        pd.DataFrame: Veritabanından yüklenen veriler bir DataFrame olarak döner.
    """
    try:
        logging.info(f"'{language}' dili için veritabanından veri yükleme işlemi başlatıldı.")
        if language == 'en':
            data = await sync_to_async(list)(UploadedEnglishTexts.objects.all())
        elif language == 'tr':
            data = await sync_to_async(list)(UploadedTurkishTexts.objects.all())
        else:
            raise ValueError("Desteklenmeyen dil kodu")

        if not data:
            logging.warning(f"'{language}' dili için veritabanında veri bulunamadı.")
            return pd.DataFrame()

        df = pd.DataFrame([model_instance.__dict__ for model_instance in data])
        df.drop(columns=['_state'], inplace=True)
        logging.info(f"'{language}' dili için veriler başarıyla yüklendi.")
        return df

    except Exception as e:
        logging.error(f"Veritabanından veri yükleme sırasında hata oluştu: {e}")
        return pd.DataFrame()