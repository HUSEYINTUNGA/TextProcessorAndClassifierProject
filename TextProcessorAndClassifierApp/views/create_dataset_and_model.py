import os
import joblib
import logging
import asyncio
import datetime
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from django.shortcuts import render, redirect
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from django.http import HttpResponse, JsonResponse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from TextProcessorAndClassifierApp.base_options import load_data_from_db
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from TextProcessorAndClassifierApp.base_options import clean_text, detect_language
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

user_csv_file = None
app_dataset = None
target_column = None
global_model_paths = {}
selected_model_config = {}

def choice_model_components(request):
    global selected_model_config,app_dataset,user_csv_file,target_column
    if request.method == 'POST':
        print('choice_model_components fonksiyonu çalıştı')
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

        if app_dataset is not None:
            selected_model_config['dataset_choice'] = 'app'
        elif user_csv_file is not None:
            selected_model_config['dataset_choice'] = 'custom'
        else:
            return JsonResponse({'error': 'Herhangi bir veri seti bulunamadı.'}, status=400)
        print('train_model fonksiyonu çağrıldı')
        model_metrics=train_model()
        return render(request, 'user_model_training_results.html',{'training_results':model_metrics})
    return render(request, 'createModelAndDataset.html')

def train_model():
    global global_model_paths, selected_model_config,target_column
    print('train_model fonksiyonu çalıştı')
    try:
        if selected_model_config.get('dataset_choice') == 'app':
            data = app_dataset
        elif selected_model_config.get('dataset_choice') == 'custom':
            data = user_csv_file
            if data is None or data.empty:
                raise ValueError("Kullanıcı dataseti bulunamadı veya boş.")
        else:
            raise ValueError("Geçersiz dataset seçimi.")
        print('dataset seçildi')
        if data.empty:
            raise ValueError("Seçilen dataset boş.")

        if target_column not in data.columns:
            target_col = 'category'
        else:
            target_col = target_column
        print("hedef sütunun kontrolü yapıldı")
        options = {
            'remove_punctuation': True,
            'remove_special_chars': True,
            'convert_to_lowercase': True,
            'remove_stopwords': True,
            'stemming': False,
            'lemmatization': True,
            }
        print("veriler temizenmeye başlıyor..")
        data['cleaned_text'] = data['text'].apply(lambda x: clean_text(x,options,language='en'))
        print("veriler temizlendi")
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
        print("vectorizer belirlendi : ",vectorizer)
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
        print("model eğitime başlıyor")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("model eğitildi")
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{algorithm}_{timestamp}"

        model_dir = "UserFiles/UserModels"
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

            return HttpResponse(request,'createModelAndDataset.html',{'choice_model_components':True})

        return redirect('upload_predict_csv') 


def select_using_dataset(request):
    """
    Kullanıcıya uygulama veri setini mi yoksa kendi veri setini mi kullanmak istediğini seçtiren fonksiyon.

    İşlevi:
    - Kullanıcının seçimine göre uygulama veri setini yükler veya kullanıcıya kendi veri setini yükleme seçeneği sunar.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Seçim sayfası veya hata mesajı.
    """
    global app_dataset
    if request.method == 'POST':
        dataset_option = request.POST.get('dataset_option', None)
        if dataset_option == 'app_dataset':
            language = request.POST.get('language', None)
            if language:
                try:
                    app_dataset = asyncio.run(load_data_from_db(language=language))
                    return render(request, 'createModelAndDataset.html', {'dataset_loaded': True, 'use_custom_dataset': False})
                except Exception as e:
                    return HttpResponse(f"Dataset yüklenirken hata oluştu: {e}", status=500)
            else:
                return JsonResponse({'error': 'Lütfen bir CSV dosyası yükleyin.'}, status=400)
        elif dataset_option == 'custom_dataset':
            print("Kullanıcı kendi datasetini yüklemek istedi")
            return render(request, 'createModelAndDataset.html', {'use_custom_dataset': True})
        else:
            return HttpResponse("Geçerli bir seçim yapılmadı.", status=400)
    return render(request, 'createModelAndDataset.html')

def custom_upload_dataset(request):
    """
    Kullanıcının kendi CSV dosyasını yüklemesini sağlayan fonksiyon.

    İşlevi:
    - Kullanıcının yüklediği CSV dosyasının formatını kontrol eder.
    - Dosyayı okuyarak kullanılabilir hale getirir.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Sütun bilgilerini içeren seçim sayfası veya hata mesajı.
    """
    global user_csv_file
    try:
        if request.method == 'POST' and request.FILES.get('csv_file'):
            csv_file = request.FILES['csv_file']

            if not csv_file.name.endswith('.csv'):
                return JsonResponse({'error': 'Lütfen bir CSV dosyası yükleyin.'}, status=400)

            try:
                user_csv_file = pd.read_csv(csv_file)
                columns = user_csv_file.columns.tolist()
                return render(request, 'createModelAndDataset.html', {
                    'use_custom_dataset': False,
                    'columns': columns,
                    'message': 'CSV dosyası başarıyla yüklendi.'
                })
            except Exception as e:
                logging.error(f"CSV dosyası işlenirken hata: {e}")
                return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=400)

        return render(request, 'createModelAndDataset.html', {'use_custom_dataset': True})
    except Exception as e:
        logging.error(f"custom_dataset fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})

def select_target_column(request):
    """
    Kullanıcının veri setinden hedef sütunu seçmesini sağlayan fonksiyon.

    İşlevi:
    - Kullanıcının seçtiği sütunun doğruluğunu kontrol eder.
    - Hedef sütunu belirleyerek sınıf dağılımını hesaplar.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Hedef sütun ve sınıf dağılımını içeren seçim sayfası veya hata mesajı.
    """
    global user_csv_file, target_column
    try:
        if request.method == 'POST':
            choise_target_column = request.POST.get('selected_column', None)
            print("Seçilen sütun", choise_target_column)
            if not choise_target_column:
                columns = user_csv_file.columns.tolist()
                return render(request, 'createModelAndDataset.html', {
                    'columns': columns,
                    'error_message': 'Hiçbir sütun seçilmedi. Lütfen en az bir sütun seçin.'
                })
            target_column = choise_target_column
            class_distribution = user_csv_file[target_column].value_counts().to_dict()
            return render(request, 'createModelAndDataset.html', {
                'selected_column': target_column,
                'columns': None,
                'balance_step': True,
                'class_distribution': class_distribution,
            })
        return render(request, 'createModelAndDataset.html')
    except Exception as e:
        logging.error(f"select_columns fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})

def balance_dataset(request):
    """
    Kullanıcının veri setindeki sınıfları dengelemesini sağlayan fonksiyon.

    İşlevi:
    - Veri setindeki sınıfların dağılımını kontrol eder.
    - Kullanıcının seçimine göre sınıf dağılımını dengeler veya olduğu gibi bırakır.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Dengelenmiş veri seti bilgilerini içeren sayfa veya hata mesajı.
    """
    global user_csv_file, target_column
    try:
        if request.method == 'POST':
            action = request.POST.get('action', None)
            if not action:
                return JsonResponse({'error': 'Bir seçenek seçmelisiniz.'}, status=400)

            target_column = target_column
            class_distribution = user_csv_file[target_column].value_counts().to_dict()

            if action == 'remove_low_classes':
                average_count = sum(class_distribution.values()) / len(class_distribution)
                user_csv_file = user_csv_file[user_csv_file[target_column].map(
                    lambda x: class_distribution.get(x, 0) >= average_count
                )]
            elif action == 'balance_distribution':
                min_count = min(class_distribution.values())
                user_csv_file = pd.concat([
                    user_csv_file[user_csv_file[target_column] == cls].sample(n=min_count, random_state=42)
                    for cls in class_distribution.keys()
                ])
            elif action == 'use_as_is':
                user_csv_file=user_csv_file
            else:
                return JsonResponse({'error': 'Geçersiz seçenek.'}, status=400)
            updated_class_distribution = user_csv_file[target_column].value_counts().to_dict()
            print(user_csv_file)
            return render(request, 'createModelAndDataset.html', {
                'updated_class_distribution': updated_class_distribution,
                'columns': None,
                'selected_column': None,
                'class_distribution': None,
                'dataset_loaded':True
            })
        else:
            return JsonResponse({'error': 'Yalnızca POST istekleri destekleniyor.'}, status=405)
    except Exception as e:
        logging.error(f"balance_dataset fonksiyonunda hata: {e}")
        return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=500)
