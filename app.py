import gradio as gr
import faiss
import json
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import os

# -----------------------------
# 1. Device setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2. FAISS Index Options
# -----------------------------
FAISS_OPTIONS = {
    "MiniLM": (
       "faiss_indices/semantic_chunks.faiss",   # FAISS index
        "chunks_semantic.json",    # clean Q–A JSON
        "sentence-transformers/all-MiniLM-L6-v2"
    )
}

loaded_indices = {}

def load_index(choice):
    """Load FAISS index, chunks, and correct retriever"""
    if choice in loaded_indices:
        return loaded_indices[choice]

    idx_path, chunk_path, model_name = FAISS_OPTIONS[choice]
    index = faiss.read_index(idx_path)

    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    retriever = SentenceTransformer(model_name, device=device)
    loaded_indices[choice] = (index, chunks, retriever)
    print(f"✅ Loaded {choice} index with {len(chunks)} clean Q–A pairs")
    return index, chunks, retriever

# -----------------------------
# 3. Generator
# -----------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=0 if device == "cuda" else -1
)

# -----------------------------
# 4. Answer Function
# -----------------------------
def answer_question(question, retriever_choice="MiniLM", k=5, threshold=0.45):
    index, chunks, retriever = load_index(retriever_choice)

    # Step A: Direct fuzzy match (more tolerant than substring)
    for entry in chunks:
        q_text = entry["question"].lower().strip("?")
        if q_text in question.lower() or question.lower() in q_text:
            return entry["answer"], f"Matched FAQ (direct): {entry['question']}"

    # Step B: Semantic search with FAISS
    q_emb = retriever.encode([question], normalize_embeddings=True)
    scores, idxs = index.search(q_emb, k)

    # Get candidates
    candidates = [chunks[i] for i in idxs[0]]
    cand_texts = [f"Q: {c['question']}\nA: {c['answer']}" for c in candidates]

    # Rerank candidates
    cand_embs = retriever.encode(cand_texts, convert_to_tensor=True, normalize_embeddings=True)
    q_emb_tensor = torch.tensor(q_emb, device=cand_embs.device)
    rerank_scores = util.cos_sim(q_emb_tensor, cand_embs)[0]

    # Pick best
    best_score, best_idx = torch.max(rerank_scores, dim=0)
    best = candidates[best_idx]

    # Threshold check: if similarity too low, fallback message
    if best_score < threshold:
        return "I could not find that in JioPay help data", ""

    context = f"Q: {best['question']}\nA: {best['answer']}"

    # Step C: Generator for natural answer
    prompt = (
        "You are JioPay-Bot. Answer the question using ONLY the context below. "
        "If the answer is not in context, say 'I could not find that in JioPay help data.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    output = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False,
        num_beams=2
    )[0]["generated_text"]

    return output, context

# -----------------------------
# 5. Gradio Interface
# -----------------------------
with gr.Blocks(title="JioPay RAG Chat Bot", theme=gr.themes.Soft()) as iface:
    gr.Markdown("## 💳 JioPay RAG Chat Bot\nAsk me anything about JioPay")

    with gr.Row():
        inp = gr.Textbox(
            lines=3,
            placeholder="Type your JioPay question here...",
            label="Your Question"
        )
        retriever_choice = gr.Dropdown(
            choices=list(FAISS_OPTIONS.keys()),
            value="MiniLM",
            label="Retriever Model"
        )

    with gr.Row():
        btn = gr.Button("🔍 Get Answer")
        clear_btn = gr.Button("🗑️ Clear", variant="secondary")

    out = gr.Textbox(lines=8, label="🤖 Answer", interactive=False)
    ctx_out = gr.Textbox(lines=6, label="Retrieved Context", interactive=False)

    btn.click(fn=answer_question, inputs=[inp, retriever_choice], outputs=[out, ctx_out])
    clear_btn.click(lambda: ("", "", ""), outputs=[inp, out, ctx_out])

if __name__ == "__main__":
    iface.launch()
