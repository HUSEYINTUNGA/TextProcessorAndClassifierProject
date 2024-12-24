document.addEventListener('DOMContentLoaded', () => {
    const textArea = document.querySelector('textarea[name="text"]');
    const classifyCheckbox = document.querySelector('input[name="classify_text"]');
    const errorContainer = document.getElementById('errorContainer');
    const hiddenInput = document.getElementById('detectedLanguage'); // Hidden input'u seç
    const supportedLanguages = ['en', 'tr']; // Desteklenen diller
    let detectedLang = null; // Tespit edilen dil burada saklanır
    let debounceTimer;

    // Kullanıcı metin girerken dil tespiti yap (debounce ile optimize edilmiş)
    textArea.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            const userText = textArea.value.trim();

            if (userText === '') {
                detectedLang = null;
                errorContainer.style.display = 'none'; // Hata mesajını gizle
                hiddenInput.value = ''; // Hidden input'u sıfırla
                return;
            }

            // Dil tespiti
            detectedLang = detectLanguage(userText);
            hiddenInput.style.display='block'; 
            hiddenInput.value = detectedLang;
        },100); 
    });

    // Sınıflandırma checkbox'ı seçildiğinde dil kontrolü yap
    classifyCheckbox.addEventListener('change', () => {
        if (classifyCheckbox.checked) {
            if (!detectedLang) {
                displayError('Lütfen metin girin ve tekrar deneyin.');
                hiddenInput.value = ''; // Hidden input'u sıfırla
                classifyCheckbox.checked = false; // Checkbox'ı otomatik devre dışı bırak
                return;
            }

            if (!supportedLanguages.includes(detectedLang)) {
                const errorMessage = `Model metnin dilini desteklemiyor: ${detectedLang}. Desteklenen diller: English (en), Türkçe (tr).`;
                hiddenInput.style.display='block';
                hiddenInput.value = errorMessage; // Hidden input'a hata mesajını yaz
                classifyCheckbox.checked = false; // Checkbox'ı otomatik devre dışı bırak
            } else {
                errorContainer.style.display = 'none'; // Hata mesajını gizle
                hiddenInput.value = `Tespit edilen dil: ${detectedLang}`; // Hidden input'a dil bilgisini yaz
                alert(`Dil tespit edildi: ${detectedLang}`);
            }
        }
    });
});

function validateAndProcess(event) {
    event.preventDefault();

    const textArea = document.querySelector('textarea[name="text"]');
    const checkboxes = document.querySelectorAll('.checkbox-group input[type="checkbox"]');
    const errorContainer = document.getElementById('errorContainer');
    errorContainer.style.display = 'none';

    if (textArea.value.trim() === '') {
        displayError('Lütfen bir metin girin.');
        return;
    }

    let isChecked = Array.from(checkboxes).some(checkbox => checkbox.checked);
    if (!isChecked) {
        displayError('Lütfen en az bir işlem seçin.');
        return;
    }

    const isUpperChecked = document.querySelector('input[name="convert_to_uppercase"]').checked;
    const isLowerChecked = document.querySelector('input[name="convert_to_lowercase"]').checked;

    if (isUpperChecked && isLowerChecked) {
        displayError('Lütfen sadece bir işlem seçin: Büyük Harfe Dönüştür veya Küçük Harfe Dönüştür.');
        return;
    }

    processAllData();

    textArea.value = '';
    checkboxes.forEach(checkbox => checkbox.checked = false);
}

function processAllData() {
    const form = document.getElementById('textProcessorForm');
    const formData = new FormData(form);

    fetch(form.action, {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Bir hata oluştu.');
        }
        return response.json();
    })
    .then(data => {
        console.log('Sunucudan dönen veri:', data);

        if (data.detected_language) {
            console.log('Dil Tespiti:', data.detected_language);
            alert(`Tespit edilen dil: ${data.detected_language}`);
        }

        if (data.processed_text) {
            console.log('İşlenmiş Metin:', data.processed_text);
            document.querySelector('.processed-text').value = data.processed_text;
            document.querySelector('.processed-text-container').style.display = 'block';

            if (data.classification_result) {
                console.log('Sınıflandırma Sonucu:', data.classification_result);
                document.getElementById('classificationText').innerText = data.classification_result;
                document.querySelector('.classification-result').style.display = 'block';
            }
        } else {
            displayError(data.error || 'Bir hata oluştu.');
        }
    })
    .catch(error => {
        console.error('Fetch sırasında bir hata oluştu:', error);
        displayError(error.message);
    });
}

function displayError(message) {
    const errorContainer = document.getElementById('errorContainer');
    errorContainer.innerText = message;
    errorContainer.style.display = 'block';
}

function copyToClipboard() {
    const processedText = document.querySelector('.processed-text');
    navigator.clipboard.writeText(processedText.value)
        .then(() => {
            const notification = document.getElementById('notification');
            notification.style.display = 'block';

            setTimeout(() => { 
                notification.style.display = 'none'; 
                const processedTextContainer = document.querySelector('.processed-text-container');
                processedTextContainer.style.display = 'none';
                const classificationResult = document.querySelector('.classification-result');
                classificationResult.style.display = 'none';
            }, 2000);
        })
        .catch(err => {
            console.error('Panoya kopyalanamadı:', err);
            alert('Metin panoya kopyalanamadı. Lütfen tekrar deneyin.');
        });
}

function detectLanguage(text) {
    const lang = window.franc(text);
    console.log("Tespit edilen metin dili:", lang);
    return lang;
}
// import { franc } from 'franc';
// function detectLanguage(text){
//     const lang = franc(text);
//     console.log("Tespit edilen metin dili:", lang);
//     return lang;
// }