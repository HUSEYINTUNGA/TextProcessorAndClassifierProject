import os
import re
import django
import string
import asyncio
import simplemma
import pandas as pd
from sklearn.svm import SVC
from joblib import dump, load
from nltk.corpus import stopwords
from asgiref.sync import sync_to_async
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TextProcessorAndClassifierProject.settings')
django.setup()
from TextProcessorAndClassifierApp.models import UploadedEnglishTexts, UploadedTurkishTexts

async def load_data_from_db(language='english'):
    """
    Veritabanından tüm verileri çeker. Dil parametresine göre İngilizce veya Türkçe veri çeker.
    
    Parameters:
    language (str): 'english' veya 'turkish' olarak dil seçimi.
    
    Returns:
    pd.DataFrame: Veritabanından çekilen metin ve kategori verileri içeren DataFrame.
    """
    os.makedirs("ML_Model", exist_ok=True)
    try:
        if language == 'english':
            data = await sync_to_async(
                lambda: list(
                    UploadedEnglishTexts.objects.all()
                    .values('text', 'category')
                )
            )()
        elif language == 'turkish':
            data = await sync_to_async(
                lambda: list(
                    UploadedTurkishTexts.objects.all()
                    .values('text', 'category')
                )
            )()
        df = pd.DataFrame(data)
        if df.empty:
            print("Veri boş!")
        return df
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return pd.DataFrame()


def clean_data(text, language='english'):
    """
    Verileri, modelin işleyebileceği formata getirmek için gerekli
    veri temizleme işlemlerini gerçekleştirir.
    
    Parameters:
    text (str): Temizlenecek metin.
    language (str): 'english' veya 'turkish' olarak dil seçimi.

    Returns:
    str: Temizlenmiş metin.
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[#$@{}\[\]/\\)(<>|!\'^+%&/\u00bd=*&\u20ac~\u00a8\u00b4\u00e6\u00a3\u00e9\u00df]', '', text)
    text = text.lower()

    if language == 'english':
        stop_words = set(stopwords.words('english'))
    elif language == 'turkish':
        stop_words = set(stopwords.words('turkish'))

    text = ' '.join([word for word in text.split() if word not in stop_words])

    if language == 'english':
        lemmatizer = WordNetLemmatizer()
        cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    else:
        cleaned_text = ' '.join([simplemma.lemmatize(word, lang='tr') for word in text.split()])

    return cleaned_text


def train_and_save_model(language='english'):
    """
    Verilerle model eğitimi gerçekleştirir ve eğitilen modeli disk üzerine kaydeder.
    
    Parameters:
    language (str): 'english' veya 'turkish' olarak dil seçimi.

    Returns:
    None: Model ve vektörizer dosyaları diske kaydedilir.
    """
    data = asyncio.run(load_data_from_db(language=language))
    if data.empty:
        print("Yetersiz veri ile model eğitimi yapılamaz.")
        return

    data['cleaned_text'] = data['text'].apply(lambda x: clean_data(x, language=language))

    X = data['cleaned_text']
    y = data['category']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Doğruluk:", accuracy)
    print("Sınıflandırma Raporu:\n", report)

    model_dir = "ML_Model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{language}_model.joblib")
    vectorizer_path = os.path.join(model_dir, f"{language}_vectorizer.joblib")
    dump(model, model_path)
    dump(vectorizer, vectorizer_path)
    print(f"Model ve vektörizer kaydedildi: {model_path}, {vectorizer_path}")

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

def predict_class(text, language='en'):
    """
    Girdi metnini sınıflandırır ve tahmin edilen sınıfı döndürür.
    
    Parameters:
    text (str): Sınıflandırılacak metin.
    language (str): 'en' veya 'tr' olarak dil seçimi.

    Returns:
    str: Tahmin edilen sınıf ismi.
    """
    if language == 'en':
        model_path = os.path.join("ML_Model", "english_model.joblib")
        vectorizer_path = os.path.join("ML_Model", "english_vectorizer.joblib")
        class_map = {v: k for k, v in convertEnglish.items()}
    elif language == 'tr':
        model_path = os.path.join("ML_Model", "turkish_model.joblib")
        vectorizer_path = os.path.join("ML_Model", "turkish_vectorizer.joblib")
        class_map = {v: k for k, v in convertTurkish.items()}
    else:
        raise ValueError("Desteklenmeyen dil kodu")

    try:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = load(model_path)
            vectorizer = load(vectorizer_path)
        else:
            raise FileNotFoundError("Model veya vektörizer bulunamadı.")

        cleaned_text = clean_data(text, language)
        text_vector = vectorizer.transform([cleaned_text])
        predicted_class_int = model.predict(text_vector)[0]

        predicted_class_name = class_map.get(predicted_class_int, "Bilinmeyen Sınıf")
        return predicted_class_name
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return "Tahmin Hatası"

