from pathlib import Path
import subprocess

repo_url = "https://github.com/encode/httpx.git"
repo_path = Path("httpx")


if not repo_path.exists():
    print("Repositório HTTPX não encontrado.")
    print("Clonando repositório...")

    subprocess.run(
        ["git", "clone", repo_url, str(repo_path)],
        check=True
    )

    print("Repositório clonado com sucesso!")
else:
    print("Repositório HTTPX já existe.")

     # Checkout do commit exigido
    
    commit = "b5addb64f0161ff6bfe94c124ef76f6a1fba5254"

print("Selecionando o commit da avaliação...")

subprocess.run(
    [
        "git",
        "-C",
        str(repo_path),
        "checkout",
        commit
    ],
    check=True
)

print("Commit da avaliação carregado com sucesso!")


    #documentação 

docs_path = repo_path / "docs"

markdown_files = list(docs_path.rglob("*.md"))

print(f"Arquivos encontrados: {len(markdown_files)}")

# Validar quantidade de Arquivos

if len(markdown_files) != 23:
    raise RuntimeError(
        f"Esperados 23 arquivos Markdown, "
        f"mas foram encontrados {len(markdown_files)}."
    )

print("Quantidade esperada de arquivos Markdown encontrada!")


documents = []

for file in markdown_files:
    try:
        text = file.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        documents.append(
            {
                "source": str(file),
                "content": text
            }
        )

    except Exception as e:
        print(f"Erro em {file}: {e}")

print(f"Documentos carregados: {len(documents)}")

# DIVIDIR EM CHUNKS

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = []

for doc in documents:

    secao_atual = "Seção não identificada"
    texto_atual = ""

    for linha in doc["content"].splitlines():

        # Se encontrar um título Markdown
        if linha.startswith("#"):
            
            # Processa o texto acumulado da seção anterior
            if texto_atual.strip():

                partes = splitter.split_text(texto_atual)

                for parte in partes:
                    chunks.append(
                        {
                            "content": parte,
                            "source": doc["source"],
                            "section": secao_atual
                        }
                    )

            # Atualiza a seção
            secao_atual = linha.lstrip("#").strip()

            texto_atual = ""

        else:
            texto_atual += linha + "\n"

    # Processar o restante do documento
    if texto_atual.strip():

        partes = splitter.split_text(texto_atual)

        for parte in partes:
            chunks.append(
                {
                    "content": parte,
                    "source": doc["source"],
                    "section": secao_atual
                }
            )
print("Total de chunks:", len(chunks))



# EMBEDDINGS

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# TRANSFORMAR EM DOCUMENTOS LANGCHAIN

from langchain_core.documents import Document

docs_langchain = []

for chunk in chunks:
    docs_langchain.append(
    Document(
        page_content=chunk["content"],
        metadata={
            "source": chunk["source"],
            "section": chunk["section"]
        }
    )
)


# BANCO VETORIAL FAISS

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    docs_langchain,
    embeddings
)

vectorstore.save_local("faiss_index")

print("Índice criado com sucesso!")



# CONFIGURAÇÃO DA LLM

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)



# PERGUNTAS PARA O RAG


while True:

    pergunta = input("\nPergunta: ")

    if pergunta.lower() == "sair":
        break

    # Busca os 3 chunks mais relevantes
    resultados = vectorstore.similarity_search_with_score(
        pergunta,
        k=3
    )

    # Montar contexto
    contexto = "\n\n".join(
        [doc.page_content for doc, score in resultados]
    )

    
    # PROMPT
    

    prompt = f"""
Você é um assistente especializado na documentação HTTPX.

Responda SEMPRE em português do Brasil.

Utilize somente as informações fornecidas no contexto abaixo.

Não invente informações que não estejam presentes no contexto.

Se a resposta não estiver no contexto, responda:

"Não encontrei essa informação na documentação disponível."

Mantenha nomes de classes, métodos, funções, parâmetros e trechos de código
exatamente como aparecem na documentação original.

Contexto:
{contexto}

Pergunta:
{pergunta}

Responda em português do Brasil:
"""

    # RESPOSTA DA LLM
   
    resposta = llm.invoke(prompt)

    print("\nResposta:\n")
    print(resposta.content)

    
    # RESULTADOS DA BUSCA
  
    print("\n" + "=" * 70)
    print("RESULTADOS DA BUSCA")
    print("=" * 70)

    for i, (doc, score) in enumerate(resultados, start=1):

        print(f"\nResultado #{i}")
        print(f"Ranking: {i}")
        print(f"Score: {score:.4f}")
        print(f"Arquivo: {doc.metadata['source']}")
        print(f"Seção: {doc.metadata.get('section', 'Não identificada')}")

        print("\nTrecho recuperado:")
        print(doc.page_content)

        print("-" * 70)