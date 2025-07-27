from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq

app = Flask(__name__)

# 🔧 Load model & data once (at startup)
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading embeddings & chunks...")
embeddings = np.load('embeddings.npy')
with open('chunks.txt', 'r', encoding='utf-8') as f:
    content = f.read()
chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]

print("Connecting to Groq API...")
client = Groq(api_key="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")

def semantic_search(user_question, top_k=100):
    q_emb = embedding_model.encode([user_question])[0]
    scores = np.dot(embeddings, q_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_idx]

def rerank_chunks(user_question, candidate_chunks):
    prompt = f"""
User question:
{user_question}

Candidate chunks:
{chr(10).join(candidate_chunks)}

Please pick the 20 most directly relevant chunks (return as plain text, keep original Urdu & Arabic).
"""
    completion = client.chat.completions.create(
        model="kimi-k2-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.strip().split('\n\n')

def generate_final_answer(user_question, relevant_chunks):
    prompt = f"""
سوال:
{user_question}

مندرجہ ذیل حوالہ جاتی متون کی مدد سے اردو میں تحقیقی اور حوالہ جاتی جواب دیں۔
متون:
{chr(10).join(relevant_chunks)}

جواب:
"""
    completion = client.chat.completions.create(
        model="kimi-k2-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.strip()

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_q = data.get("question")
    if not user_q:
        return jsonify({"error": "Missing 'question' in JSON"}), 400

    print(f"Received question: {user_q}")

    try:
        top_chunks = semantic_search(user_q, top_k=100)
        relevant_chunks = rerank_chunks(user_q, top_chunks)
        answer = generate_final_answer(user_q, relevant_chunks)
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("✅ Server starting on 0.0.0.0:8080...")
    app.run(host="0.0.0.0", port=8080)
