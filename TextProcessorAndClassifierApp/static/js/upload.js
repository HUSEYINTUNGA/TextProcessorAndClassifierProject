document.addEventListener("DOMContentLoaded", () => {
    // CSV yükleme formu, dosya yükleme butonu ve dosya bilgisi alanları
    const csvForm = document.getElementById("csv-form");
    const uploadButton = document.getElementById("upload-button");
    const fileInfoDiv = document.getElementById("file-info");
    const fileNameSpan = document.getElementById("selected-file-name");
    const columnsForm = document.getElementById("columns-form");
    const operationForm = document.getElementById("operation-form");

    // Hata mesajlarını göstermek için yardımcı fonksiyon
    const showError = (element, message) => {
        const errorElement = element.querySelector(".error-message");
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.style.display = "block"; // Hata mesajını göster
        }
    };

    // Hata mesajlarını temizlemek için yardımcı fonksiyon
    const clearError = (element) => {
        const errorElement = element.querySelector(".error-message");
        if (errorElement) {
            errorElement.textContent = "";
            errorElement.style.display = "none"; // Hata mesajını gizle
        }
    };

    // CSV formu yüklendiğinde, dosya kontrolü ve hata mesajı gösterimi
    if (csvForm) {
        csvForm.addEventListener("submit", (event) => {
            const fileInput = document.getElementById("csvFile");

            // Dosya seçilmediğinde hata mesajı göster
            if (!fileInput.value) {
                showError(csvForm, "Lütfen bir dosya seçin.");
                event.preventDefault();
            } else {
                const file = fileInput.files[0];
                const fileName = file.name;
                const fileSize = file.size;

                // Dosya uzantısının .csv olduğundan emin ol
                if (!fileName.endsWith(".csv")) {
                    showError(csvForm, "Lütfen yalnızca .csv uzantılı dosyalar yükleyin.");
                    event.preventDefault();
                }
                // Dosya boyutunun 10MB'den küçük olduğundan emin ol
                else if (fileSize > 10 * 1024 * 1024) {
                    showError(csvForm, "Dosya boyutu çok büyük. Lütfen 10MB'den küçük bir dosya yükleyin.");
                    event.preventDefault();
                } else {
                    // Dosya doğruysa, hata mesajlarını temizle ve dosya bilgilerini göster
                    clearError(csvForm);
                    fileNameSpan.textContent = fileName;
                    fileInfoDiv.style.display = "block";

                }
            }
        });
    }
    // Sütun seçimi formu gönderildiğinde işlem formunu göster
    if (columnsForm) {
        columnsForm.addEventListener("submit", (event) => {
            event.preventDefault(); // Formun sayfa yenilemesini engelle
            const selectedColumns = Array.from(
                document.querySelectorAll('input[name="selected_columns"]:checked')
            ).map((checkbox) => checkbox.value);

            // Sütun seçilmediyse kullanıcıyı bilgilendir
            if (selectedColumns.length === 0) {
                showError(columnsForm, "Lütfen en az bir sütun seçin.");
            } else {
                clearError(columnsForm); // Hata mesajını temizle

                // İşlem seçim formunu göster
                if (operationForm) {
                    operationForm.style.display = "block";
                }

                // Sütun seçimi formunu gizle
                columnsForm.style.display = "none";
            }
        });
    }

    // İşlem seçim formu ve seçilen işlemi göndermek için buton
    const submitOperationButton = document.getElementById("submit-operation");
    if (submitOperationButton) {
        submitOperationButton.addEventListener("click", () => {
            const selectedOperation = document.querySelector('input[name="operation"]:checked');

            // Eğer işlem seçilmediyse kullanıcıyı bilgilendir
            if (!selectedOperation) {
                showError(operationForm, "Lütfen bir işlem seçin.");
            } else {
                clearError(operationForm); // Hata mesajını temizle

                // Seçilen işleme göre yönlendirme URL'sini belirle
                const operation = selectedOperation.value;
                const redirectUrl = operation === "process"
                    ? "{% url 'process_text' %}"
                    : "{% url 'predict_class' %}";

                window.location.href = ${redirectUrl}?columns=${JSON.stringify(selectedColumns)};
            }
        });
    }
});