document.addEventListener("DOMContentLoaded", () => {
    const selectColumnsForm = document.getElementById("select-columns-form");
    const columnsError = document.getElementById("columns-error");
    const fileInput = document.getElementById("csvFile");
    const fileError = document.getElementById("file-error");

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            const file = fileInput.files[0];

            if (file.size > 10 * 1024 * 1024) {
                fileError.textContent = "Dosya boyutu çok büyük! Lütfen 10 MB'den küçük bir dosya yükleyin.";
                fileError.style.display = "block";
                fileInput.value = ""; 
            } else {
                fileError.style.display = "none";
            }
        });
    }

    if (selectColumnsForm) {
        selectColumnsForm.addEventListener("submit", (event) => {
            const selectedColumns = Array.from(
                selectColumnsForm.querySelectorAll('input[name="selected_columns"]:checked')
            );

            if (selectedColumns.length === 0) {
                event.preventDefault();
                columnsError.textContent = "Lütfen en az bir sütun seçin.";
                columnsError.style.display = "block";
            } else {
                columnsError.style.display = "none";
            }
        });
    }
});

