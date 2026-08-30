import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="RAG - HTTPX",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TÍTULO
# ============================================================

st.title("📚 RAG - Documentação HTTPX")

st.write(
    "Sistema de Retrieval-Augmented Generation (RAG) "
    "utilizando a documentação do HTTPX."
)

st.divider()


# ============================================================
# CARREGAR EMBEDDINGS
# ============================================================

@st.cache_resource
def carregar_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# CARREGAR FAISS
# ============================================================

@st.cache_resource
def carregar_vectorstore():

    embeddings = carregar_embeddings()

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


# ============================================================
# CARREGAR LLM
# ============================================================

@st.cache_resource
def carregar_llm():

    return ChatOllama(
        model="llama3.2",
        temperature=0
    )


# ============================================================
# CARREGAMENTO DOS COMPONENTES
# ============================================================

try:

    with st.spinner("Carregando banco vetorial..."):

        vectorstore = carregar_vectorstore()

    st.success("Banco vetorial FAISS carregado com sucesso!")

except Exception as e:

    st.error("Erro ao carregar o banco vetorial.")

    st.code(str(e))

    st.stop()


try:

    llm = carregar_llm()

    st.success("Modelo Llama 3.2 carregado com sucesso!")

except Exception as e:

    st.error("Erro ao carregar o modelo Llama 3.2.")

    st.code(str(e))

    st.stop()


st.divider()


# ============================================================
# CAMPO DE PERGUNTA
# ============================================================

st.subheader("🔎 Fazer uma pergunta")

pergunta = st.text_input(
    "Digite sua pergunta sobre o HTTPX:",
    placeholder="Ex.: Como criar um cliente no HTTPX?"
)


# ============================================================
# BOTÃO CONSULTAR
# ============================================================

if st.button("🔎 Consultar", type="primary"):

    if not pergunta.strip():

        st.warning("Digite uma pergunta antes de consultar.")

    else:

        try:

            # ====================================================
            # BUSCA SEMÂNTICA
            # ====================================================

            with st.spinner("Buscando informações na documentação..."):

                resultados = vectorstore.similarity_search_with_score(
                    pergunta,
                    k=3
                )


            # ====================================================
            # CONTEXTO
            # ====================================================

            contexto = "\n\n".join(
                [
                    doc.page_content
                    for doc, score in resultados
                ]
            )


            # ====================================================
            # PROMPT
            # ====================================================

            prompt = f"""
Você é um assistente especializado na documentação HTTPX.

Responda SEMPRE em português do Brasil.

Utilize somente as informações fornecidas no contexto abaixo.

Não invente informações que não estejam presentes no contexto.

Se a resposta não estiver no contexto, responda:

"Não encontrei essa informação na documentação disponível."

Mantenha nomes de classes, métodos, funções, parâmetros e
trechos de código exatamente como aparecem na documentação original.

Contexto:
{contexto}

Pergunta:
{pergunta}

Responda em português do Brasil:
"""


            # ====================================================
            # LLM
            # ====================================================

            with st.spinner("Gerando resposta..."):

                resposta = llm.invoke(prompt)


            # ====================================================
            # RESPOSTA
            # ====================================================

            st.divider()

            st.subheader("💬 Resposta")

            st.write(resposta.content)


            # ====================================================
            # RESULTADOS RECUPERADOS
            # ====================================================

            st.divider()

            st.subheader("📄 Documentos recuperados")

            for i, (doc, score) in enumerate(
                resultados,
                start=1
            ):

                with st.expander(
                    f"Resultado #{i} — Score: {score:.4f}"
                ):

                    st.write(
                        f"**Ranking:** {i}"
                    )

                    st.write(
                        f"**Score:** {score:.4f}"
                    )

                    st.write(
                        f"**Arquivo:** "
                        f"{doc.metadata.get('source', 'Não informado')}"
                    )

                    st.write(
                        f"**Seção:** "
                        f"{doc.metadata.get('section', 'Não identificada')}"
                    )

                    st.markdown(
                        "**Trecho recuperado:**"
                    )

                    st.code(
                        doc.page_content
                    )


        except Exception as e:

            st.error("Ocorreu um erro durante a consulta.")

            st.code(str(e))

