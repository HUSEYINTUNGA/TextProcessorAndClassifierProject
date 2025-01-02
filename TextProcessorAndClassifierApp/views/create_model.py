from django.shortcuts import render, redirect
from django.http import JsonResponse
from TextProcessorAndClassifierApp.base_options import clean_text, detect_language
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib
import os
import logging
import datetime

global_model_paths = {}
selected_model_config = {}
user_csv_file = None
app_dataset = None
target_column = None

def choice_model_components(request):
    global selected_model_config,app_dataset,user_csv_file,target_column
    if request.method == 'POST':
        selected_algorithm = request.POST.get('algorithm', 'SVM')
        selected_vectorizer = request.POST.get('vectorizer', 'TF-IDF')
        c_value = request.POST.get('c_value', '1.0')
        max_depth = request.POST.get('max_depth', None)
        n_estimators = request.POST.get('n_estimators', '100')
        n_neighbors = request.POST.get('n_neighbors', '5')
        test_size = request.POST.get('test_size', '0.2')
        random_state = request.POST.get('random_state', '42')

        selected_model_config = {
            'selected_algorithm': selected_algorithm,
            'selected_vectorizer': selected_vectorizer,
            'c_value': c_value,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'n_neighbors': n_neighbors,
            'test_size': test_size,
            'random_state': random_state
        }

        app_dataset = request.shared_data['app_dataset']
        user_csv_file = request.shared_data['user_csv_file']
        target_column = request.shared_data['target_column']
        print(request.shared_data.get('user_csv_file', None))
        if app_dataset is not None:
            selected_model_config['dataset_choice'] = 'app'
        elif user_csv_file is not None:
            selected_model_config['dataset_choice'] = 'custom'
        else:
            return JsonResponse({'error': ' Bu mesajı choice_model_components fonksiyonu gönderdi. Herhangi bir veri seti bulunamadı.'}, status=400)
        model_metrics=train_model()
        return render(request, 'user_model_training_results.html',{'training_results':model_metrics})
    return render(request, 'createModel.html')

def train_model():
    global global_model_paths, selected_model_config,target_column

    try:
        if selected_model_config.get('dataset_choice') == 'app':
            data = app_dataset
        elif selected_model_config.get('dataset_choice') == 'custom':
            data = user_csv_file
            if data is None or data.empty:
                raise ValueError("Kullanıcı dataseti bulunamadı veya boş.")
        else:
            raise ValueError("Geçersiz dataset seçimi.")

        if data.empty:
            raise ValueError("Seçilen dataset boş.")

        if target_column not in data.columns:
            target_col = 'category'
        else:
            target_col = target_column

        data['cleaned_text'] = data.apply(
            lambda row: clean_text(
                " ".join([str(row[col]) for col in data.columns if col != target_col]),
                language=detect_language(" ".join([str(row[col]) for col in data.columns if col != target_col]))
            ),
            axis=1
        )
        X = data['cleaned_text']
        y = data[target_col]

        vectorizer_type = selected_model_config['selected_vectorizer']
        if vectorizer_type == 'TF-IDF':
            vectorizer = TfidfVectorizer()
        elif vectorizer_type == 'Count':
            vectorizer = CountVectorizer()
        elif vectorizer_type == 'Hashing':
            vectorizer = HashingVectorizer()
        else:
            raise ValueError("Desteklenmeyen vektörleştirici türü.")

        X = vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(selected_model_config.get('test_size')), random_state=int(selected_model_config.get('random_state'))
        )
        algorithm = selected_model_config['selected_algorithm']
        if algorithm == 'SVM':
            model = SVC(C=float(selected_model_config['c_value']))
        elif algorithm == 'Random Forest':
            model = RandomForestClassifier(
                max_depth=int(selected_model_config['max_depth']) if selected_model_config['max_depth'] else None,
                n_estimators=int(selected_model_config['n_estimators'])
            )
        elif algorithm == 'Logistic Regression':
            model = LogisticRegression(C=float(selected_model_config['c_value']))
        elif algorithm == 'Naive Bayes':
            model = MultinomialNB()
        elif algorithm == 'Decision Tree':
            model = DecisionTreeClassifier(max_depth=int(selected_model_config['max_depth']) if selected_model_config['max_depth'] else None)
        elif algorithm == 'KNN':
            model = KNeighborsClassifier(n_neighbors=int(selected_model_config['n_neighbors']))
        elif algorithm == 'Gradient Boosting':
            model = GradientBoostingClassifier()
        else:
            raise ValueError("Desteklenmeyen algoritma türü.")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{algorithm}_{timestamp}"

        model_dir = "UserCsvFiles/UserModels"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
        vectorizer_path = os.path.join(model_dir, f"{model_name}_vectorizer.joblib")

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
        logging.error(f"Eğitim sırasında bir hata oluştu: {str(e)}")
        raise Exception(f"Model eğitimi sırasında bir hata oluştu: {str(e)}")

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

            return redirect('choice_model_components')

        return redirect('upload_predict_csv') 
