import os
import joblib
import logging
import asyncio
import pandas as pd
from sklearn.svm import SVC
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from django.shortcuts import render, redirect
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from django.http import HttpResponse, JsonResponse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from TextProcessorAndClassifierApp.base_options import clean_text, detect_language
from TextProcessorAndClassifierApp.base_options import load_data_from_db, predict_class
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

csv_file_for_training = None
app_dataset_for_training = None
target_column_of_training_dataset = None
generated_model_path=None
generated_vectorizer_path=None
selected_model_config = {}
csv_file_to_be_classified = None
selected_columns = None


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
    global app_dataset_for_training
    if request.method == 'POST':
        dataset_option = request.POST.get('dataset_option', None)
        if dataset_option == 'app_dataset':
            language = request.POST.get('language', None)
            if language:
                try:
                    app_dataset_for_training = asyncio.run(load_data_from_db(language=language))
                    return render(request, 'choiceTrainDataset.html', {'dataset_loaded': True, 'use_custom_dataset': False})
                except Exception as e:
                    return HttpResponse(f"Dataset yüklenirken hata oluştu: {e}", status=500)
            else:
                return JsonResponse({'error': 'Lütfen bir CSV dosyası yükleyin.'}, status=400)
        elif dataset_option == 'custom_dataset':
            return render(request, 'choiceTrainDataset.html', {'use_custom_dataset': True})
        else:
            return HttpResponse("Geçerli bir seçim yapılmadı.", status=400)
    return render(request, 'choiceTrainDataset.html')

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
    global csv_file_for_training
    try:
        if request.method == 'POST' and request.FILES.get('csv_file'):
            csv_file = request.FILES['csv_file']

            if not csv_file.name.endswith('.csv'):
                return JsonResponse({'error': 'Lütfen bir CSV dosyası yükleyin.'}, status=400)

            try:
                csv_file_for_training = pd.read_csv(csv_file)
                columns = csv_file_for_training.columns.tolist()
                return render(request, 'choiceTrainDataset.html', {
                    'use_custom_dataset': False,
                    'columns': columns,
                    'message': 'CSV dosyası başarıyla yüklendi.'
                })
            except Exception as e:
                logging.error(f"CSV dosyası işlenirken hata: {e}")
                return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=400)

        return render(request, 'choiceTrainDataset.html', {'use_custom_dataset': True})
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
    global csv_file_for_training, target_column_of_training_dataset
    try:
        if request.method == 'POST':
            choise_target_column = request.POST.get('selected_column', None)
            if not choise_target_column:
                columns = csv_file_to_be_classified.columns.tolist()
                return render(request, 'choiceTrainDataset.html', {
                    'columns': columns,
                    'error_message': 'Hiçbir sütun seçilmedi. Lütfen en az bir sütun seçin.'
                })
            target_column_of_training_dataset = choise_target_column
            class_distribution = csv_file_for_training[target_column_of_training_dataset].value_counts().to_dict()
            return render(request, 'choiceTrainDataset.html', {
                'selected_column': target_column_of_training_dataset,
                'columns': None,
                'balance_step': True,
                'class_distribution': class_distribution,
            })
        return render(request, 'choiceTrainDataset.html')
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
    global csv_file_for_training, target_column_of_training_dataset
    try:
        if request.method == 'POST':
            action = request.POST.get('action', None)
            if not action:
                return JsonResponse({'error': 'Bir seçenek seçmelisiniz.'}, status=400)

            target_column_of_training_dataset = target_column_of_training_dataset
            class_distribution = csv_file_for_training[target_column_of_training_dataset].value_counts().to_dict()

            if action == 'remove_low_classes':
                average_count = sum(class_distribution.values()) / len(class_distribution)
                csv_file_for_training = csv_file_for_training[csv_file_for_training[target_column_of_training_dataset].map(
                    lambda x: class_distribution.get(x, 0) >= average_count
                )]
            elif action == 'balance_distribution':
                min_count = min(class_distribution.values())
                csv_file_for_training = pd.concat([
                    csv_file_for_training[csv_file_for_training[target_column_of_training_dataset] == cls].sample(n=min_count, random_state=42)
                    for cls in class_distribution.keys()
                ])
            elif action == 'use_as_is':
                csv_file_for_training=csv_file_for_training
            else:
                return JsonResponse({'error': 'Geçersiz seçenek.'}, status=400)
            updated_class_distribution = csv_file_for_training[target_column_of_training_dataset].value_counts().to_dict()
            return render(request, 'choiceTrainDataset.html', {
                'updated_class_distribution': updated_class_distribution,
                'columns': None,
                'selected_column': None,
                'class_distribution': None,
            })
        else:
            return JsonResponse({'error': 'Yalnızca POST istekleri destekleniyor.'}, status=405)
    except Exception as e:
        logging.error(f"balance_dataset fonksiyonunda hata: {e}")
        return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=500)



def choice_model_components(request):
    global selected_model_config,app_dataset_for_training,csv_file_for_training
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

        if app_dataset_for_training is not None:
            selected_model_config['dataset_choice'] = 'app'
        elif csv_file_for_training is not None:
            selected_model_config['dataset_choice'] = 'custom'
        else:
            return JsonResponse({'error': 'Herhangi bir veri seti bulunamadı.'}, status=400)
        model_metrics=train_model()
        return render(request, 'user_model_training_results.html',{'training_results':model_metrics})
    return render(request, 'createModel.html')

def train_model():
    global generated_model_path, generated_vectorizer_path, selected_model_config, target_column_of_training_dataset, csv_file_for_training, app_dataset_for_training

    try:
        if selected_model_config.get('dataset_choice') == 'app':
            data = app_dataset_for_training
        elif selected_model_config.get('dataset_choice') == 'custom':
            data = csv_file_for_training
            if data is None or data.empty:
                raise ValueError("Kullanıcı dataseti bulunamadı veya boş.")
        else:
            raise ValueError("Geçersiz dataset seçimi.")

        if data.empty:
            raise ValueError("Seçilen dataset boş.")

        if target_column_of_training_dataset not in data.columns:
            target_col = 'category'
        else:
            target_col = target_column_of_training_dataset

        options = {
            'remove_punctuation': True,
            'remove_special_chars': True,
            'convert_to_lowercase': True,
            'remove_stopwords': True,
            'stemming': False,
            'lemmatization': True,
            }

        data['cleaned_text'] = data['text'].apply(lambda x: clean_text(x,options,language='en'))
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{algorithm}_{timestamp}"

        model_dir = "UserFiles/UserModels"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
        vectorizer_path = os.path.join(model_dir, f"{model_name}_vectorizer.joblib")

        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)

        generated_model_path=model_path
        generated_vectorizer_path=vectorizer_path

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
    global generated_model_path, generated_vectorizer_path

    if request.method == 'POST':
        user_choice = request.POST.get('user_choice')

        if user_choice not in ['yes', 'no']:
            return JsonResponse({'error': 'Geçersiz seçim.'}, status=400)

        if user_choice == 'no':
            
            if generated_model_path and generated_vectorizer_path:
                try:
                    os.remove(generated_model_path)
                    os.remove(generated_vectorizer_path)
                except FileNotFoundError:
                    pass 


            return render(request, 'createModel.html')


        return redirect('upload_predict_csv')

    return JsonResponse({'error': 'Yalnızca POST isteği desteklenir.'}, status=405)



def upload_predict_csv_file(request):
    """
    Kullanıcıdan CSV dosyasını yüklemesini isteyen ve bu dosyanın sütunlarını döndüren fonksiyon.

    İşlevi:
    - Kullanıcının yüklediği CSV dosyasının formatını doğrular.
    - Dosyadaki sütunları analiz ederek kullanıcıya seçim için sunar.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Sütun bilgilerini içeren sınıflandırma sayfası veya hata mesajı.
    """
    global csv_file_to_be_classified
    try:
        if request.method == 'POST' and request.FILES.get('csv_file'):
            csv_file = request.FILES['csv_file']

            if not csv_file.name.endswith('.csv'):
                return JsonResponse({'error': 'Lütfen bir CSV dosyası yükleyin.'}, status=400)

            try:
                csv_file_to_be_classified = pd.read_csv(csv_file)
                columns = csv_file_to_be_classified.columns.tolist()
                return render(request, 'predictCsvFile.html', {'columns': columns, 'message': 'CSV dosyası başarıyla yüklendi.'})
            except Exception as e:
                logging.error(f"CSV dosyası işlenirken hata: {e}")
                return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=400)

        return render(request, 'predictCsvFile.html')
    except Exception as e:
        logging.error(f"classify_csv_page fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})

def select_predict_columns(request):
    """
    Kullanıcının seçtiği sütunları işleyerek global değişkene kaydeden fonksiyon.

    İşlevi:
    - Kullanıcının seçtiği sütunların geçerliliğini kontrol eder.
    - Seçilen sütunları daha sonraki işlemler için global değişkene kaydeder.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: Seçim sayfası veya hata mesajı.
    """
    global csv_file_to_be_classified, selected_columns
    try:
        if request.method == 'POST':
            selected_columns = request.POST.getlist('selected_columns')

            if not selected_columns:
                columns = csv_file_to_be_classified.columns.tolist()
                return render(request, 'predictCsvFile.html', {
                    'columns': columns,
                    'error_message': 'Hiçbir sütun seçilmedi. Lütfen en az bir sütun seçin.'
                })

            return render(request, 'predictCsvFile.html', {
                'selected_columns': selected_columns,
            })

        return render(request, 'predictCsvFile.html')
    except Exception as e:
        logging.error(f"select_columns fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'})

def classifier_csv_file(request):
    """
    Kullanıcının seçtiği sütunlarda sınıf tahmini yapar ve işlenmiş CSV dosyasını döndüren fonksiyon.

    İşlevi:
    - Orijinal metin verilerini işlenmek üzere hazırlar.
    - Metinleri temizlemek ve işlemek için `clean_text` fonksiyonunu kullanır.
    - İşlenmiş verilerle dil tespiti ve sınıf tahmini yapar.
    - Tahmin yapılamayan verileri işaretler.
    - İşlenmiş ve sınıflandırılmış CSV dosyasını kullanıcıya sunar.

    Parametreler:
        request (HttpRequest): Kullanıcıdan gelen istek.

    Döndürür:
        HttpResponse: İşlenmiş CSV dosyasını veya hata mesajı.
    """
    global csv_file_to_be_classified, selected_columns, generated_model_path, generated_vectorizer_path

    try:
        if csv_file_to_be_classified is None:
            return JsonResponse({'error': 'CSV dosyası yüklenmemiş.'}, status=400)

        if not selected_columns:
            return JsonResponse({'error': 'Sütun seçilmemiş. Lütfen önce sütun seçimi yapın.'}, status=400)

        for column in selected_columns:
            if column not in csv_file_to_be_classified.columns:
                return JsonResponse({'error': f"'{column}' sütunu bulunamadı."}, status=400)

            predicted_classes = []

            for row in csv_file_to_be_classified[column]:
                original_text = str(row)

                detected_language = detect_language(original_text)

                processed_text = clean_text(original_text, {
                    'remove_punctuation': True,
                    'remove_special_chars': True,
                    'convert_to_lowercase': True,
                    'remove_stopwords': True,
                    'stemming': False,
                    'lemmatization': True
                }, detected_language)

                try:
                    predicted_class = predict_class(processed_text,generated_model_path,generated_vectorizer_path,detected_language)
                except Exception as e:
                    logging.error(f"Tahmin sırasında hata: {e}")
                    predicted_classes.append("Tahmin yapılamadı")
                    continue

                predicted_classes.append(predicted_class)

            result_column_name = f"{column}_class"
            csv_file_to_be_classified[result_column_name] = predicted_classes

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_filename = f"classifiedCsv_{timestamp}.csv"
        output_path = os.path.join('UserFiles/ClassifyCsvFiles', output_filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        csv_file_to_be_classified.to_csv(output_path, index=False)

        with open(output_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{output_filename}"'
            return response

    except Exception as e:
        logging.error(f"classify_csv_data fonksiyonunda hata: {e}")
        return JsonResponse({'error': 'Bir hata oluştu. Lütfen tekrar deneyin.'}, status=500)