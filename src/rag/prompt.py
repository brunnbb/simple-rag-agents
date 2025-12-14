from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

MODEL_CONTEXT = """
Você é um consultor agrícola especializado em produção e manejo de maçãs,
com foco em ajudar produtores rurais, considerando o contexto produtivo
da região sul do Brasil. Forneça respostas claras, objetivas e práticas,
em parágrafo único, sem markdown ou formatação.
"""

RAG_PROMPT = PromptTemplate.from_template(
"""
{system_context}

Contexto recuperado do banco vetorial:
{context}

Pergunta:
{question}
"""
)

async def build_rag_prompt(question: str, retriever) -> str:
    docs: list[Document] = await retriever.ainvoke(question)

    context = "\n\n".join(d.page_content for d in docs)
    if not context.strip():
        context = "Nenhum contexto encontrado no banco vetorial."

    return RAG_PROMPT.format(
        system_context=MODEL_CONTEXT,
        context=context,
        question=question,
    )
