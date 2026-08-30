# RAG sobre a Documentação HTTPX

# Identificação

* Nome do aluno: Gabriel Lucas de Oliveira Costa
* Formato da solução: script Python
* Link do vídeo: 
* Link do Colab, se aplicável: Não se aplica.


## Objetivo

Este projeto implementa um sistema de RAG utilizando a documentação do HTTPX como base de conhecimento.

O sistema realiza automaticamente a obtenção do repositório HTTPX, seleciona um commit específico para garantir a reprodutibilidade, localiza os arquivos Markdown da pasta `docs/`, divide os documentos em chunks, gera embeddings semânticos, cria um índice vetorial utilizando FAISS e recupera os trechos mais relevantes para responder às perguntas.

A geração das respostas é realizada pelo modelo Llama 3.2, executado localmente por meio do Ollama.

O sistema é instruído a utilizar somente as informações recuperadas da documentação e a informar quando a resposta não estiver presente no contexto disponível.


## Arquitetura resumida

O fluxo principal do sistema pode ser representado da seguinte forma:


Repositório HTTPX
       ↓
Checkout do commit específico
       ↓
Localização dos arquivos .md
       ↓
Leitura dos documentos
       ↓
Divisão em chunks
       ↓
Geração de embeddings
       ↓
Índice vetorial FAISS
       ↓
Busca por similaridade
       ↓
Recuperação dos 3 chunks mais relevantes
       ↓
Construção do prompt
       ↓
Llama 3.2 via Ollama
       ↓
Resposta em português


O fluxo resumido do RAG é:

documentos → chunks → embeddings → índice/busca → resultados → geração


# Como executar do zero

# 1. Versão aproximada do Python

O projeto foi desenvolvido para ser executado com **Python 3.10 ou superior**.

Também é necessário possuir o Git instalado para realizar a clonagem do repositório HTTPX.

Para utilizar a geração de respostas, é necessário instalar o Ollama.

### 2. Dependências

As principais bibliotecas utilizadas pelo projeto são:

* `langchain`
* `langchain-community`
* `langchain-core`
* `langchain-text-splitters`
* `langchain-huggingface`
* `langchain-ollama`
* `sentence-transformers`
* `faiss-cpu`
* `streamlit`

A instalação pode ser realizada com:

```bash
pip install langchain
pip install langchain-community
pip install langchain-core
pip install langchain-text-splitters
pip install langchain-huggingface
pip install langchain-ollama
pip install sentence-transformers
pip install faiss-cpu
pip install streamlit
```

### 3. Como obter a base HTTPX

O sistema utiliza o repositório oficial do HTTPX:

```text
https://github.com/encode/httpx.git
```

O próprio script verifica se o diretório `httpx/` existe.

Caso não exista, o repositório é clonado automaticamente.

Depois da clonagem, o sistema realiza o checkout do commit utilizado na avaliação:

```text
b5addb64f0161ff6bfe94c124ef76f6a1fba5254
```

O código utiliza:

```python
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
```

A documentação utilizada pelo sistema está localizada em:

```text
httpx/docs/
```

Os arquivos Markdown são encontrados recursivamente por meio de:

```python
markdown_files = list(docs_path.rglob("*.md"))
```

O sistema também valida se foram encontrados exatamente 23 arquivos Markdown.

### 4. Qual arquivo executar

O arquivo principal do projeto é:

```text
rag_httpx.py

Mas também foi inserido uma interface.

app.py
```

Para executar:

```bash
python rag_httpx.py
```

Durante a execução, o sistema:

1. Verifica se o repositório HTTPX existe.
2. Clona o repositório caso necessário.
3. Seleciona o commit definido para a avaliação.
4. Localiza os arquivos Markdown dentro de `httpx/docs/`.
5. Valida a quantidade de arquivos encontrados.
6. Lê os documentos.
7. Divide os documentos em chunks.
8. Gera os embeddings.
9. Cria o índice vetorial FAISS.
10. Inicializa o modelo Llama 3.2 através do Ollama.
11. Aguarda uma pergunta do usuário.

Após a criação do índice, caso utilize a interface web pode ser iniciada com:

streamlit run app.py

O arquivo app.py carrega o índice FAISS existente, recebe as perguntas do usuário pela interface Streamlit, recupera os três chunks mais relevantes e envia o contexto recuperado para o modelo Llama 3.2 através do Ollama.

A aplicação pode ser acessada localmente em:

http://localhost:8501

### 5. Como fazer uma pergunta

Após executar o programa, será exibido:

```text
Pergunta:
```

Um exemplo de pergunta é:

```text
Como criar um Client no HTTPX?
```

Para encerrar o programa:

```text
sair
```

---

# Decisões técnicas

# Chunking

* Estratégia: `RecursiveCharacterTextSplitter`, do LangChain.
* Tamanho aproximado: 500 caracteres.
* Overlap: 100 caracteres.
* Justificativa: a divisão dos documentos em trechos menores facilita a recuperação de partes específicas da documentação. O overlap de 100 caracteres ajuda a preservar informações próximas às fronteiras entre chunks.

Antes da divisão, o código identifica títulos Markdown e utiliza a seção correspondente como metadado do chunk.

A configuração utilizada é:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

---

# Embeddings e busca

* Modelo de embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
* Tecnologia utilizada: Hugging Face Embeddings.
* Índice vetorial: FAISS.
* Forma de busca:`similarity_search_with_score()`.
* Valor de `top_k`: 3.
* Justificativa: são recuperados os três chunks considerados mais relevantes para a pergunta, fornecendo ao modelo um contexto reduzido e diretamente relacionado à consulta.

A busca é realizada com:

```python
resultados = vectorstore.similarity_search_with_score(
    pergunta,
    k=3
)
```

Os resultados recuperados são utilizados para construir o contexto enviado ao modelo Llama 3.2.

---

# Metadados e fontes

Cada chunk mantém informações sobre sua origem.

O sistema preserva:

* Caminho do arquivo: armazenado no campo `source`.
* Seção da documentação: armazenada no campo `section`.

Essas informações são adicionadas aos documentos LangChain:

```python
Document(
    page_content=chunk["content"],
    metadata={
        "source": chunk["source"],
        "section": chunk["section"]
    }
)
```

Durante a recuperação, o sistema exibe:

```text
Ranking
Score
Arquivo
Seção
Trecho recuperado
```

Dessa forma, é possível identificar de qual arquivo e seção da documentação veio cada trecho utilizado para responder à pergunta.

# Perguntas de teste

# 1. Pergunta com resposta clara

* Pergunta:

```text
Como criar um Client no HTTPX?
```

* Uma resposta esperada pode indicar que um cliente pode ser criado utilizando httpx.Client(), por exemplo:

client = httpx.Client()

Esse cliente pode ser utilizado para realizar requisições HTTP.

O resultado foi relevante? Por quê?

Sim. O resultado foi considerado relevante porque os chunks recuperados estavam diretamente relacionados ao Client do HTTPX e forneceram informações utilizadas pela LLM para construir a resposta. Além disso, os resultados exibidos pelo sistema permitem identificar o arquivo, a seção e o trecho da documentação utilizado na recuperação.

---

# 2. Pergunta ampla ou ambígua

* Pergunta:

```text
Como o HTTPX funciona?
```

* Resultado esperado: O HTTPX funciona roteando requisições com base em esquema, domínio, porta ou uma combinação desses.

Em outras palavras, o HTTPX permite que você especifique como o cliente deve rotear a requisição para o servidor, utilizando os seguintes critérios:

    Esquema (http ou https)
    Domínio
    Porta

Com isso, você pode especificar como o cliente deve rotear a requisição para o servidor, utilizando os critérios acima.

* O resultado foi relevante? Por quê?

Parcialmente, porque as informações apresentadas estão relacionadas à documentação do HTTPX e foram recuperadas a partir dos documentos utilizados como base de conhecimento.

Entretanto, a resposta não apresenta uma explicação completa sobre como o HTTPX funciona, pois a pergunta é ampla e o sistema recupera somente os 3 chunks mais relevantes. Dessa forma, os trechos selecionados podem representar apenas alguns aspectos específicos da documentação, como roteamento e extensões, em vez de fornecer uma visão geral do funcionamento da biblioteca.



# 3. Pergunta fora do escopo

* **Pergunta:**

```text
Qual é a capital do Brasil?
```

* Como o sistema reagiu:

O prompt instrui a LLM a responder:

```text

Não encontrei essa informação na documentação disponível.

caso a informação solicitada não esteja presente no contexto recuperado.

* Como essa reação poderia melhorar:

O sistema poderia utilizar uma etapa adicional de verificação de relevância antes de enviar os resultados para a LLM. Dessa forma, perguntas claramente fora do domínio poderiam ser identificadas antes da geração da resposta.


# Limitações conhecidas

* O sistema depende da qualidade dos chunks recuperados pela busca vetorial.
* Perguntas muito amplas ou ambíguas podem recuperar informações que não sejam suficientes para responder completamente à pergunta.
* O sistema utiliza apenas os três chunks mais relevantes (`top_k=3`).
* O modelo Llama 3.2 depende dos recursos disponíveis no computador para execução local.
* O Ollama precisa estar instalado e o modelo `llama3.2` precisa estar disponível localmente.
* O projeto depende de conexão com a internet na etapa de clonagem do repositório HTTPX.
* A quantidade de arquivos Markdown é validada como 23; alterações futuras no repositório podem fazer essa validação falhar caso o corpus seja diferente.
* O sistema foi desenvolvido especificamente para responder perguntas relacionadas à documentação do HTTPX.
* A busca vetorial pode recuperar resultados mesmo quando uma pergunta está fora do domínio, pois a decisão final depende do contexto recuperado e das instruções fornecidas à LLM.

---

# Uso de ferramentas de IA

* Ferramentas utilizadas: modelos de inteligência artificial utilizados como apoio durante o desenvolvimento e revisão do projeto.
* Tarefas em que ajudaram: apoio na compreensão dos conceitos de RAG, embeddings, busca vetorial, LangChain, FAISS e integração com modelos de linguagem.
* Exemplo representativo de prompt ou orientação:

```text
Como implementar um sistema RAG utilizando a documentação do HTTPX,
com divisão dos documentos em chunks, embeddings, FAISS e geração
de respostas utilizando uma LLM local?
```

* O que foi testado, modificado ou validado pelo aluno:

O código foi implementado e testado para realizar a clonagem do repositório HTTPX, selecionar o commit especificado, localizar os arquivos Markdown, dividir os documentos em chunks, gerar embeddings, criar o índice FAISS, realizar a busca por similaridade e gerar respostas utilizando o Llama 3.2 através do Ollama.

Também foram realizadas validações da quantidade de arquivos Markdown e dos resultados recuperados pela busca.


# Referências e código externo

Usei como referência  video aulas explicando a realização de um Projeto RAG passo a passo para maior compreensão prática e conceitual.

Aqui está alguns exemplos de referência utilizada.

https://medium.com/data-hackers/construindo-aplica%C3%A7%C3%B5es-personalizadas-com-llm-atrav%C3%A9s-de-rag-retrieve-augmented-generation-6f3a3df7b6de

https://www.youtube.com/watch?v=0M8iO5ykY-E

https://www.youtube.com/watch?v=yPz8LDcAcdA

Video exemplo para criação do Ambiente virtual para inicialização do projeto 

https://www.youtube.com/watch?v=wOchmO8J7gA&t=502s

https://dev.to/asouza/como-implementar-um-sistema-rag-do-zero-em-python-1ej7

https://www.hashtagtreinamentos.com/sistema-rag-python


# Documentação HTTPX

Repositório oficial utilizado como base documental:

```text
https://github.com/encode/httpx
```

Commit utilizado:

```text
b5addb64f0161ff6bfe94c124ef76f6a1fba5254

A documentação utilizada encontra-se em:

httpx/docs/
```

# Tecnologias e bibliotecas

* Python
* LangChain
* LangChain Text Splitters
* LangChain Hugging Face
* LangChain Ollama
* FAISS
* Sentence Transformers
* Hugging Face
* Ollama
* Git
* Streamlit

O código utiliza bibliotecas de terceiros para implementar as etapas de divisão de documentos, geração de embeddings, armazenamento vetorial e comunicação com o modelo de linguagem.


# Segurança

* [x] Minha solução não usa API key.
* [ ] Minha solução usa segredo protegido e nenhuma chave foi publicada.
