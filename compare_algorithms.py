import asyncio
import os
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from asgiref.sync import sync_to_async
import django
import simplemma  
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




def compare_algorithms(language='english'):
    """
    Verilerle farklı sınıflandırma algoritmalarını eğitir ve performanslarını karşılaştırır.
    
    Parameters:
    language (str): 'english' veya 'turkish' olarak dil seçimi.

    Returns:
    None: Algoritmaların doğruluk ve sınıflandırma raporları çıktı olarak verilir.
    """
    data = asyncio.run(load_data_from_db(language=language))
    if data.empty:
        print("Yetersiz veri ile algoritmalar karşılaştırılamaz.")
        return

    data['cleaned_text'] = data['text'].apply(lambda x: clean_data(x, language=language))

    X_raw = data['cleaned_text']
    y = data['category']

    vectorizers = {
        'TfidfVectorizer': TfidfVectorizer(),
        'CountVectorizer': CountVectorizer(),
        'HashingVectorizer': HashingVectorizer(n_features=5000)
    }

    algorithms = {
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Logistic Regression': LogisticRegression(),
        'SVC': SVC(),
    }

    for vec_name, vectorizer in vectorizers.items():
        print(f"\n{vec_name} ile vektörleştirme")
        X = vectorizer.fit_transform(X_raw)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for name, model in algorithms.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            print(f"\nAlgoritma: {name}")
            print("Doğruluk:", accuracy)
            print("Sınıflandırma Raporu:\n", report)


compare_algorithms('turkish')
