# Financial & Payroll Data RAG System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline designed to extract general ledger, accrual, and payroll batch data from an Oracle Database, vectorize it using Azure OpenAI, store it in a ChromaDB vector database, and provide an interactive terminal interface for domain-specific Q&A.

---

## Configuration Setup

Before running the application, you must configure your environment and local files.

### 1. External Configuration (`config.py`)

Create a file named `config.py` in the same directory as your script and specify the following variables:

```python
# Oracle DB Credentials
ORACLE_USER = "your_oracle_username"
ORACLE_PASSWORD = "your_oracle_password"
ORACLE_CONNECT_STRING = "your_oracle_connection_string_or_dsn"

# ChromaDB Configurations
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "payroll_accrual_collection"

# Azure OpenAI Deployment Names
AZURE_EMBEDDING_DEPLOYMENT = "your_embedding_model_deployment_name"
AZURE_CHAT_DEPLOYMENT = "your_chat_model_deployment_name"

```

### 2. Environment Variables

Set up your Azure OpenAI credentials in your system environment:

```bash
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"

```

### 3. Oracle Instant Client

Ensure you update the initialization line in the script with your local machine's Oracle Instant Client directory path:

```python
oracledb.init_oracle_client(lib_dir=r'/path/to/your/instantclient')

```

---

## How the Code Works (Step-by-Step)

The architecture is divided into two primary loops: **Ingestion** (Data Collection & Vectorization) and **Execution** (Query Retrieval & LLM Generation).


### Step 1: Client Initializations

* The script initializes the Oracle thick-client driver using a localized directory pathway to the Oracle Instant Client binaries.
* `get_azure_client()` reads authentication secrets directly from system environment variables to securely instantiate the `AzureOpenAI` engine wrapper.

### Step 2: Data Extraction & Document Normalization

* `fetch_data_from_oracle()` opens a secure connection to your database instance and fires a comprehensive SQL query targeting the ledger table `NATL_WORKDAY_CLOUD_GL_MAIN`.
* The resulting rows are zippered with lowercased column names into dictionaries.
* To prepare structured text for vector search, the data fields (such as `Batch Name`, `Organization`, `Amounts`, and `Trans ID`) are concatenated into a human-readable text string. All raw data key-value pairs are saved into a separate dictionary to serve as descriptive metadata.

### Step 3: Vector Embedding Generation

* `get_embeddings()` handles communication with the Azure OpenAI server.
* It sends natural text payloads to the embedding deployment and returns dense, floating-point multi-dimensional arrays (vector representations) that encapsulate the semantic meaning of each ledger record.

### Step 4: Building the ChromaDB Vector Store

* `build_vector_store()` sets up a persistent disk-backed database directory using ChromaDB.
* It implements a drop-and-rebuild strategy: checking for and deleting any pre-existing collections of the same name to avoid stale duplicate entries.
* A new collection is spun up utilizing **Cosine Similarity** (`"hnsw:space": "cosine"`) as its metric metric space.
* Data is pushed in chunks of **100 records** at a time. For every batch, text is processed into embeddings and saved directly alongside its baseline unique identifier, metadata payload, and source document string.

### Step 5: Semantic Retrieval Loop

* When a question is submitted to `query_vector_store()`, the system transforms the input string into a search vector using the same embedding pipeline.
* The system scans the ChromaDB database to perform an HNSW cosine similarity search, returning the top $k$ (defaulting to 5) records closest in meaning to the question.

### Step 6: Contextual LLM Response Generation

* `ask_llm()` aggregates the raw string bodies of the top matching documents into a single block of context.
* This context is passed to the Azure chat model using strict system prompt instructions:
> *"Use ONLY the provided context to answer. If the answer is not in the context, say so."*


* Setting the `temperature=0.2` ensures deterministic, factual responses while completely preventing hallucinations based outside the extracted DB rows.

---

## Usage Instructions

### Run Ingestion Pipeline

To fetch data from Oracle DB, vectorize it, and build your local ChromaDB vector store, run:

```bash
python rag.py ingest

```

### Launch Interactive Q&A Interface

Once ingestion is complete, run the script without arguments to enter an interactive loop directly within your terminal console:

```bash
python rag.py

```

* **To Exit:** Type `quit`, `exit`, or `q`.
