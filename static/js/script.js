
const arquivo = document.getElementById("arquivo");

const uploadArea = document.getElementById("uploadArea");

const uploadTitle = document.getElementById("uploadTitle");

const uploadText = document.getElementById("uploadText");

const form = document.getElementById("testForm");

const button = document.getElementById("generateButton");

const buttonText = document.getElementById("buttonText");

const loader = document.getElementById("loader");


// Arquivo selecionado
arquivo.addEventListener("change", function () {

    if (this.files.length === 0) {
        return;
    }

    const file = this.files[0];

    if (!file.name.toLowerCase().endsWith(".zip")) {

        alert("Selecione um arquivo ZIP.");

        this.value = "";

        return;
    }

    uploadTitle.textContent = file.name;

    uploadText.textContent = "Arquivo selecionado com sucesso";

    uploadArea.classList.add("dragover");

});


// Drag & Drop

uploadArea.addEventListener("dragover", function (event) {

    event.preventDefault();

    uploadArea.classList.add("dragover");

});


uploadArea.addEventListener("dragleave", function () {

    uploadArea.classList.remove("dragover");

});


uploadArea.addEventListener("drop", function (event) {

    event.preventDefault();

    uploadArea.classList.remove("dragover");

    const files = event.dataTransfer.files;

    if (files.length === 0) {
        return;
    }

    const file = files[0];

    if (!file.name.toLowerCase().endsWith(".zip")) {

        alert("Selecione um arquivo ZIP.");

        return;
    }

    arquivo.files = files;

    uploadTitle.textContent = file.name;

    uploadText.textContent = "Arquivo selecionado com sucesso";

    uploadArea.classList.add("dragover");

});


// Envio do formulário

form.addEventListener("submit", function () {

    button.disabled = true;

    buttonText.textContent = "Gerando testes...";

    loader.style.display = "inline-block";

});
