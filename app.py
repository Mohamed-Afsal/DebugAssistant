from flask import Flask, request, jsonify, render_template
import os
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import warnings

# Suppress pymilvus warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymilvus")

# === Flask Setup ===
app = Flask(__name__)

# === Config ===
LOG_FILE_PATH = "/root/DebugAssistant/grc.log"  # <--- Set this to your actual log file
Settings.embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1")

# IBM WatsonX credentials
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": os.getenv("WML_API_KEY")
}
project_id = os.getenv("WML_PROJECT_ID")
model_id = "mistralai/mixtral-8x7b-instruct-v01"
parameters = { GenParams.MAX_NEW_TOKENS: 2000 }

LLM_model_watsonX = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

# === Error Context Extraction ===
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

# === Load once at startup ===
print("🧠 Initializing vector index from logs...")
log_text = extract_error_context(LOG_FILE_PATH)
documents = [Document(text=log_text)]
vec_index = VectorStoreIndex.from_documents(documents, show_progress=True)
retriever = vec_index.as_retriever(similarity_top_k=6)

# === Routes ===
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "").strip()

    if not question or len(question.split()) < 2:
        return jsonify({"response": "⚠️ Please ask a more specific question."})

    top_k_nodes = retriever.retrieve(question)

    prompt = f"""
You are a debug assistant. Use the logs below to help the user understand the root cause of the issue.

=== LOGS ===
{''.join(node.text for node in top_k_nodes)}

=== QUESTION ===
{question}

Answer:"""

    answer = LLM_model_watsonX.generate_text(prompt=prompt)
    return jsonify({"response": answer.strip()})

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
