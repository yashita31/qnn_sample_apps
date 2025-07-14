from openai import OpenAI
from chromadb import PersistentClient
from embedder import Embedder

chroma_collections = {}
# Configs
MODEL_NAME = "qnn-deepseek-r1-distill-qwen-7b"
CHROMA_DIR = "./src/llm/chroma_store"
MODEL_ENDPOINT = "http://localhost:5272/v1/"
ROOT_DOC_DIR = "./src/llm/docs"

# Setup: RAG and LLM
client = OpenAI(base_url=MODEL_ENDPOINT, api_key="unused")
embedder = Embedder(model_dir="./models/mobilebert-onnx", use_cpu=False)
chroma = PersistentClient(path=CHROMA_DIR)


def retrieve_context(query: str, group: str, top_k=5):

    try:
        collection = chroma.get_collection(group)
    except Exception as e:
        print(f"Collection for group '{group}' not found. Falling back to general.")
        try:
            collection = chroma.get_collection("general")
        except:
            return ""

    embeddings = embedder.embed([query])[0]
    results = collection.query(query_embeddings=[embeddings], n_results=top_k)
    return "\n\n".join(results["documents"][0])


def query_llm_with_rag(user_query: str, group: str = "general"):
    print(f"\n[Querying LLM with RAG]\nUser Query: {user_query}\nGroup: {group}")
    context = retrieve_context(user_query, group=group)
    print(f"Retrieved Context: {context[:100]}...")
    if context:
        query_input = f"Context:\n{context}\n\nQuestion: {user_query}"
    else:
        query_input = user_query
    print(f"Query Input: {query_input}...")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": query_input}],
            },
        ],
        model=MODEL_NAME,
        max_tokens=500,
        temperature=0,
        top_p=0.9,
        stream=True,
    )
    full_response = ""
    print("\n Response:")
    for chunk in response:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            print(delta.content, end="", flush=True)
            full_response += delta.content
    return full_response


# === Main Entrypoint ===
# if __name__ == "__main__":
#     query = input("Enter your query: ")
#     answer = query_llm_with_rag(query)
#     print("\n Response:")
#     print(answer)
