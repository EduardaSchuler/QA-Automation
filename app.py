from flask import Flask, render_template, request, send_file
import os
import zipfile
import tempfile
from pathlib import Path

import AgentValidator as validator

app = Flask(__name__)

EXTENSOES = {".pdf", ".docx", ".xlsx", ".xlsm", ".iqy"}


def ler_zip(zip_path):

    pasta = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(pasta)

    contexto = ""

    for arquivo in Path(pasta).rglob("*"):

        if not arquivo.is_file():
            continue

        extensao = arquivo.suffix.lower()

        if extensao not in EXTENSOES:
            continue

        try:

            conteudo = validator.ler_arquivo(str(arquivo), extensao)

            contexto += f"\n\n### FONTE: {arquivo.name}\n" f"{conteudo}"

            print(f"Arquivo lido: {arquivo}")

        except Exception as erro:

            print(f"Erro ao ler {arquivo}: {erro}")

    return contexto


@app.route("/")
def index():

    return render_template("index.html")


@app.route("/gerar", methods=["POST"])
def gerar():

    if "arquivo" not in request.files:
        return "Nenhum arquivo foi enviado.", 400

    arquivo = request.files["arquivo"]

    if arquivo.filename == "":
        return "Nenhum arquivo foi selecionado.", 400

    if not arquivo.filename.lower().endswith(".zip"):
        return "O arquivo precisa ser um ZIP.", 400

    try:

        quantidade = int(request.form.get("quantidade", 10))

    except ValueError:

        quantidade = 10

    if quantidade < 1:
        quantidade = 1

    print()
    print("=" * 70)
    print("GERAÇÃO DE TESTES")
    print("=" * 70)
    print()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp:

        arquivo.save(temp.name)

        zip_path = temp.name

    try:

        print("Lendo arquivos do ZIP...")

        contexto = ler_zip(zip_path)

        if not contexto.strip():

            return (
                "Nenhum arquivo suportado foi encontrado "
                "ou nenhum conteúdo pôde ser extraído.",
                400,
            )

        print()
        print(f"Caracteres extraídos: {len(contexto)}")

        print()
        print(f"Gerando {quantidade} perguntas...")

        testes = validator.gerar_perguntas_respostas(
            contexto=contexto, quantidade=quantidade
        )

        if not testes:

            return ("O LLM não retornou nenhum caso de teste.", 500)

        print(f"Testes gerados: {len(testes)}")

        caminho_saida = os.path.join(tempfile.gettempdir(), "testes_gerados.txt")

        validator.salvar_testes_txt(testes, caminho_saida)

        print()
        print("=" * 70)
        print("TESTES GERADOS COM SUCESSO")
        print("=" * 70)
        print()

        return send_file(
            caminho_saida, as_attachment=True, download_name="testes_gerados.txt"
        )

    finally:

        try:

            os.remove(zip_path)

        except Exception:

            pass


if __name__ == "__main__":

    app.run(debug=True)
