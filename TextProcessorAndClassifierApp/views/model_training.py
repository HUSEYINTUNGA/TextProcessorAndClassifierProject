from django.shortcuts import render, redirect
from django.http import JsonResponse
from train_models import load_data_from_db, clean_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os
import asyncio

global_model_settings = {}
global_model_paths = {}

def create_model(request):
    """
    Kullanıcının model oluşturma sayfasını görüntüler ve seçimleri işler.
    """
    global global_model_settings

    if request.method == 'POST':
        selected_algorithm = request.POST.get('algorithm', 'SVM')
        selected_vectorizer = request.POST.get('vectorizer', 'TF-IDF')
        dataset_choice = request.POST.get('dataset_choice', 'application')
        dataset_language = request.POST.get('dataset_language', 'english')

        c_value = request.POST.get('c_value', '1.0')
        max_depth = request.POST.get('max_depth', None)
        n_estimators = request.POST.get('n_estimators', '100')
        test_size = request.POST.get('test_size', '0.2')
        random_state = request.POST.get('random_state', '42')

        global_model_settings = {
            'selected_algorithm': selected_algorithm,
            'selected_vectorizer': selected_vectorizer,
            'dataset_choice': dataset_choice,
            'dataset_language': dataset_language,
            'c_value': c_value,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'test_size': test_size,
            'random_state': random_state
        }

        if dataset_choice == 'application':
            return redirect('train_model')
        else:
            return redirect('upload_training_data')

    algorithms = ['SVM', 'Random Forest', 'Logistic Regression']
    vectorizers = ['TF-IDF', 'CountVectorizer', 'HashingVectorizer']

    return render(request, 'model_creation.html', {
        'algorithms': algorithms,
        'vectorizers': vectorizers,
        'default_test_size': '0.2',
        'default_random_state': '42'
    })

def train_model(language, algorithm, vectorizer_type, params):
    """
    Model eğitim işlemini gerçekleştirir ve sonucu döndürür.
    """
    global global_model_paths
    try:
        data = asyncio.run(load_data_from_db(language=language))
        if data.empty:
            raise ValueError("Seçilen dataset boş.")

        data['cleaned_text'] = data['text'].apply(lambda x: clean_data(x, language=language))
        X = data['cleaned_text']
        y = data['category']

        if vectorizer_type == 'TF-IDF':
            vectorizer = TfidfVectorizer()
        elif vectorizer_type == 'CountVectorizer':
            vectorizer = CountVectorizer()
        else:
            raise ValueError("Desteklenmeyen vektörleştirici türü.")

        X = vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(params['test_size']), random_state=int(params['random_state'])
        )

        if algorithm == 'SVM':
            model = SVC(C=float(params['c_value']))
        elif algorithm == 'Random Forest':
            model = RandomForestClassifier(
                max_depth=params['max_depth'], 
                n_estimators=int(params['n_estimators'])
            )
        elif algorithm == 'Logistic Regression':
            model = LogisticRegression(C=float(params['c_value']))
        else:
            raise ValueError("Desteklenmeyen algoritma türü.")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        model_dir = "UserCsvFiles/UserModels"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{language}_model.joblib")
        vectorizer_path = os.path.join(model_dir, f"{language}_vectorizer.joblib")
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)

        global_model_paths = {
            'model_path': model_path,
            'vectorizer_path': vectorizer_path
        }

        return {
            'accuracy': accuracy,
            'classification_report': report
        }

    except Exception as e:
        raise Exception(f"Eğitim sırasında bir hata oluştu: {str(e)}")

def train_user_model(request):
    """
    Kullanıcının oluşturduğu modeli eğitir ve sonucu kullanıcıya sunar.
    """
    try:
        language = global_model_settings.get('dataset_language', 'en')
        algorithm = global_model_settings.get('selected_algorithm', 'SVM')
        vectorizer_type = global_model_settings.get('selected_vectorizer', 'TF-IDF')
        params = {
            'test_size': global_model_settings.get('test_size', '0.2'),
            'random_state': global_model_settings.get('random_state', '42'),
            'c_value': global_model_settings.get('c_value', '1.0'),
            'max_depth': global_model_settings.get('max_depth'),
            'n_estimators': global_model_settings.get('n_estimators', '100')
        }

        training_results = train_model(language, algorithm, vectorizer_type, params)

        return render(request, 'user_model_training_results.html', {
            'training_results': training_results,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def handle_user_satisfaction(request):
    """
    Kullanıcının memnuniyet durumuna göre işlemleri yönetir.
    """
    global global_model_paths

    if request.method == 'POST':
        user_choice = request.POST.get('user_choice')

        if user_choice not in ['yes', 'no']:
            return JsonResponse({'error': 'Geçersiz seçim.'}, status=400)

        if user_choice == 'no':

            if global_model_paths:
                try:
                    os.remove(global_model_paths['model_path'])
                    os.remove(global_model_paths['vectorizer_path'])
                except FileNotFoundError:
                    pass
                finally:
                    global_model_paths = {}

            return redirect('create_model')


        return redirect('upload_predict_csv')
