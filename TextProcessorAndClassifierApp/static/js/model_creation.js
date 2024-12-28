document.addEventListener('DOMContentLoaded', function () {
    const algorithmSelect = document.getElementById('algorithm');
    const parameterGroups = document.querySelectorAll('.parameter-group');
    const datasetLanguageOptions = document.getElementById('dataset-language-options');
    const datasetChoiceRadios = document.querySelectorAll('input[name="dataset_choice"]');

    function updateParameterVisibility() {
        parameterGroups.forEach(group => group.style.display = 'none');

        const selectedAlgorithm = algorithmSelect.value.replace(/ /g, '-');
        const activeGroup = document.getElementById(`${selectedAlgorithm}-parameters`);
        if (activeGroup) {
            activeGroup.style.display = 'block';
        }
    }

    function updateDatasetOptions() {
        const selectedChoice = document.querySelector('input[name="dataset_choice"]:checked');
        if (selectedChoice && selectedChoice.value === 'application') {
            datasetLanguageOptions.style.display = 'block';
        } else {
            datasetLanguageOptions.style.display = 'none';
        }
    }

    algorithmSelect.addEventListener('change', updateParameterVisibility);
    datasetChoiceRadios.forEach(radio => {
        radio.addEventListener('change', updateDatasetOptions);
    });

    updateParameterVisibility();
    updateDatasetOptions();
});
