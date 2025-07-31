# logs_debugger.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymilvus")

import os
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# === Settings ===
LOG_FILE_PATH = "/root/DebugAssistant/*.log"  # <-- update this with actual path

# === Embedding Model ===
Settings.embed_model = HuggingFaceEmbedding(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)

# === IBM WatsonX model config ===
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("WML_API_KEY")
}
project_id = os.getenv("WML_PROJECT_ID")
model_id = "mistralai/mixtral-8x7b-instruct-v01"
parameters = {
    GenParams.MAX_NEW_TOKENS: 2000
}
LLM_model_watsonX = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

# === Extract log error context ===
def extract_error_context(log_file, lines_before=25, lines_after=25):
    with open(log_file, 'r') as file:
        lines = file.readlines()

    error_indices = [i for i, line in enumerate(lines) if "error" in line.lower()]
    if not error_indices:
        return "No ERROR logs found."

    result = []
    start = max(error_indices[0] - lines_before, 0)
    end = min(error_indices[0] + lines_after + 1, len(lines))

    for i in range(1, len(error_indices)):
        next_start = max(error_indices[i] - lines_before, 0)
        next_end = min(error_indices[i] + lines_after + 1, len(lines))
        if next_start <= end:
            end = max(end, next_end)
        else:
            result.append(''.join(lines[start:end]))
            result.append("=" * 80)
            start, end = next_start, next_end

    result.append(''.join(lines[start:end]))
    return '\n'.join(result)

# === Core Chat Function ===
def start_debug_chat():
    text_errors = extract_error_context(LOG_FILE_PATH)
    documents = [Document(text=text_errors)]
    vec_index = VectorStoreIndex.from_documents(documents, show_progress=True)
    query_engine = vec_index.as_retriever(similarity_top_k=6)

    print("\n🧠 Debug Assistant is ready! Ask your questions (type 'exit' to quit):\n")

    while True:
        qsn = input("You: ").strip()
        if qsn.lower() == 'exit':
            print("👋 Goodbye!")
            break
        if not qsn:
            print("⚠️  Please enter a question.\n")
            continue

        top_k_nodes = query_engine.retrieve(qsn)

        prompt = (
            "You are a debug assistant. Based on the following logs, answer the user's question and help identify the root cause.\n\n"
            "Logs:\n"
        )
        for node in top_k_nodes:
            prompt += node.text + "\n"
        prompt += "\nQuestion: " + qsn

        response = LLM_model_watsonX.generate_text(prompt=prompt)
        print("\n🤖 Response:", response, "\n", "*" * 40, "\n")

# === Entry Point ===
if __name__ == "__main__":
    print("🚀 Launching Debug Assistant...\n")
    start_debug_chat()

