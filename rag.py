import oracledb
import chromadb
from openai import AzureOpenAI
import config #this file has to be created to import all the external sources used in the code

oracledb.init_oracle_client(lib_dir=r'instant client path')


def get_azure_client():
    import os
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-06-01",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )


def fetch_data_from_oracle():
    connection = oracledb.connect(
        user=config.ORACLE_USER,
        password=config.ORACLE_PASSWORD,
        dsn=config.ORACLE_CONNECT_STRING,
    )
    cursor = connection.cursor()

    cursor.execute("""
        SELECT ACCOUNTING_DATE, SOBP, ORGANIZATION, FACCT, CSUB, IET,
               FINANCIAL_PROJECT, PRODUCT_SERVICE, CURRENCY_CODE,
               ACCOUNTED_CR, ACCOUNTED_DR, ENTERED_DR, ENTERED_CR,
               BATCH_NAME, BATCH_DESCRIPTION, JOURNAL_ENTRY_DESCRIPTION,
               LINE_DESCRIPTION, SOURCE, CATEGORY, CODE_COMBINATION,
               GROUP_ID, ATTRIBUTE1, ATTRIBUTE2, ATTRIBUTE3, ATTRIBUTE4,
               ATTRIBUTE5, ATTRIBUTE6, SEGMENT1, SEGMENT2, SEGMENT3,
               SEGMENT4, SEGMENT5, SEGMENT6, SEGMENT7, SEGMENT8, SEGMENT9,
               CREATION_DATE, CREATED_BY, LAST_UPDATED_DATE, LAST_UPDATED_BY,
               STATUS, ERROR_MESSAGE, TRANS_ID, SCHEDULE_ID, FILE_NAME_REF
        FROM NATL_WORKDAY_CLOUD_GL_MAIN
    """)

    columns = [col[0].lower() for col in cursor.description]
    rows = cursor.fetchall()

    records = []
    for idx, row in enumerate(rows):
        record = dict(zip(columns, row))
        text = (
            f"Accounting Date: {record.get('accounting_date')} | "
            f"Batch Name: {record.get('batch_name')} | "
            f"Batch Description: {record.get('batch_description')} | "
            f"Journal Entry: {record.get('journal_entry_description')} | "
            f"Line Description: {record.get('line_description')} | "
            f"Organization: {record.get('organization')} | "
            f"Source: {record.get('source')} | "
            f"Category: {record.get('category')} | "
            f"Currency: {record.get('currency_code')} | "
            f"Accounted CR: {record.get('accounted_cr')} | "
            f"Accounted DR: {record.get('accounted_dr')} | "
            f"Entered DR: {record.get('entered_dr')} | "
            f"Entered CR: {record.get('entered_cr')} | "
            f"Status: {record.get('status')} | "
            f"File: {record.get('file_name_ref')} | "
            f"Code Combination: {record.get('code_combination')} | "
            f"SOBP: {record.get('sobp')} | "
            f"FACCT: {record.get('facct')} | "
            f"CSUB: {record.get('csub')} | "
            f"IET: {record.get('iet')} | "
            f"Financial Project: {record.get('financial_project')} | "
            f"Product Service: {record.get('product_service')} | "
            f"Group ID: {record.get('group_id')} | "
            f"Trans ID: {record.get('trans_id')} | "
            f"Schedule ID: {record.get('schedule_id')}"
        )
        records.append({
            "id": str(record.get("trans_id") or f"row_{idx}"),
            "text": text,
            "metadata": {k: str(v) for k, v in record.items() if v is not None},
        })

    cursor.close()
    connection.close()
    return records


def get_embeddings(texts, client):
    response = client.embeddings.create(
        input=texts,
        model=config.AZURE_EMBEDDING_DEPLOYMENT,
    )
    return [item.embedding for item in response.data]


def build_vector_store(records, client):
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    try:
        chroma_client.delete_collection(config.CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        texts = [r["text"] for r in batch]
        ids = [r["id"] for r in batch]
        metadatas = [r["metadata"] for r in batch]
        embeddings = get_embeddings(texts, client)

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"Indexed {len(records)} records into ChromaDB.")
    return collection


def query_vector_store(query, client, top_k=5):
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        print("No data in vector store. Run 'python rag.py ingest' first.")
        return []

    query_embedding = get_embeddings([query], client)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return results["documents"][0] if results["documents"] else []


def ask_llm(query, context_docs, client):
    context = "\n\n".join(context_docs)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions about "
                "accrual and payroll batch data. Use ONLY the provided context "
                "to answer. If the answer is not in the context, say so."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer based on the context above."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=config.AZURE_CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content


def ingest():
    """Full ingestion pipeline: fetch from Oracle DB -> embed -> store in ChromaDB."""
    client = get_azure_client()
    print("Fetching data from Oracle database...")
    records = fetch_data_from_oracle()
    print(f"Fetched {len(records)} records.")

    print("Building vector store...")
    build_vector_store(records, client)
    print("Ingestion complete.")


def ask(query):
    client = get_azure_client()

    print(f"Searching for: {query}")
    context_docs = query_vector_store(query, client)

    if not context_docs:
        return "No relevant data found in the knowledge base."

    print(f"Found {len(context_docs)} relevant records. Generating answer...")
    answer = ask_llm(query, context_docs, client)
    return answer


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
    else:
        print("RAG System Ready. Type 'quit' to exit.\n")
        while True:
            query = input("Enter your question: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            answer = ask(query)
            print(f"\nAnswer:\n{answer}\n")
