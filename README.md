# Simple RAG-Agents

Pipeline simples de **RAG (Retrieval-Augmented Generation)** para ingestão de documentos em **ChromaDB** e execução de perguntas contra múltiplos modelos de LLM (OpenAI, Gemini e DeepSeek), com processamento assíncrono e logging estruturado.
> **Detalhe:** todos os comandos abaixo devem ser executados **a partir da raiz do projeto**.
---

## Requisitos

- Python **3.12+**
- Chaves de API:
  - OpenAI
  - Gemini
  - DeepSeek
- [`uv`](https://github.com/astral-sh/uv) (Recomendado)

---

## Estrutura do Projeto

```
RAG-AGENT/
├── chroma_db/            # Vector database persistido
├── data/                 # JSONL de chunks e CSV de perguntas
├── logs/                 # Logs do pipeline criados em execução
├── src/
│   ├── emb_db/           # Ingestão das chunks e teste do ChromaDB
│   │   ├── ingest.py
│   │   └── test.py
│   ├── llm/              # Clientes, providers e retry
│   ├── pipeline/         # Processamento do CSV
│   ├── rag/              # Prompt e retriever
│   ├── config.py         
│   └── main.py           
├── .env
├── pyproject.toml
```

---

## Configuração do Ambiente

Crie um arquivo `.env` na raiz do projeto:

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
```

---

## Instalação das Dependências (uv)

```bash
uv sync
```

Ou, se preferir não usar o uv:

```bash
pip install -r requirements.txt
```

---

## Ingestão dos Documentos (Embeddings)

**Gera (ou recria) o banco vetorial a partir do arquivo `data/chunks_out.jsonl`:**

```bash
uv run -m src.emb_db.ingest create
```

Ou usando o python normal

```bash
python -m src.emb_db.ingest create
```

**Para adicionar novos documentos sem apagar o banco existente:**

```bash
uv run -m src.emb_db.ingest append
```

Ou usando o python normal

```bash
python -m src.emb_db.ingest append
```

---

## Teste do ChromaDB

**Testa se os embeddings foram persistidos corretamente:**
```bash
uv run -m src.emb_db.test
```

Ou usando o python normal
```bash
python -m src.emb_db.test
```

---

## Execução do Pipeline RAG

Processa o CSV de perguntas (`data/perguntas_e_respostas_rag.csv`) e gera respostas para todos os modelos configurados:

```bash
uv run -m src.main
```
Ou usando python normal

```bash
python -m src.main
```

---

## Logs

Os logs são gravados após a execução em:

```
logs/pipeline.log
```

Incluem:
- Progresso do pipeline
- Requests HTTP (via httpx)
- Erros de LLM (OpenAI, Gemini, DeepSeek)

---

## Observações

- O processamento é **assíncrono**, com controle de concorrência por semáforos.
- Cada pergunta é salva incrementalmente no CSV para evitar perda de progresso.
- O banco vetorial é persistido localmente via ChromaDB.
