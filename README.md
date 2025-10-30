# 🧠 RAG-JioPay

This report presents a comprehensive analysis of a **production-grade Retrieval-Augmented Generation (RAG) chatbot** developed for **JioPay customer support**.  

The system implements **five distinct chunking strategies**, **three embedding models**, and **three data ingestion pipelines**. Through systematic ablation studies, we evaluate the performance impact of different architectural choices on **retrieval accuracy**, **generation quality**, and **system latency**.  

Our findings reveal that **semantic chunking with E5 embeddings** achieves optimal performance, leading to its selection for the final production deployment. The system demonstrates superior retrieval precision, with semantic chunking producing **493 fine-grained chunks** for comprehensive context coverage.

---

## 🏗️ System Overview

The RAG system consists of three main components:  
- **Data Ingestion Pipeline**  
- **Retrieval System**  
- **Generation Module**

The architecture processes JioPay documentation through multiple chunking strategies, embeds text using various models, and retrieves relevant context for **LLM-based answer generation**.

---

## ⚙️ Architecture Components

- **Data Sources:** JioPay help center, business documentation, and FAQ repositories  
- **Chunking Module:** Five strategies evaluated; **Semantic chunking** selected for production  
- **Embedding System:** Three models compared; **E5** selected for optimal performance  
- **Vector Store:** **FAISS** indices for efficient similarity search  
- **Generation:** **Flan-T5 model** for context-grounded response generation  
- **Production Deployment:** [Custom Web Application](https://jiopay-flow.lovable.app)  
- **Alternative Deployment:** [Hugging Face Spaces](https://huggingface.co/spaces/Manjiri/RAG_JioPay_BOT)

---

## 🔬 Ablation Studies

| Aspect | Description |
|--------|--------------|
| **Objective** | Compare chunking, embedding, and ingestion strategies |
| **Key Finding** | Semantic chunking + E5 embeddings yielded highest retrieval accuracy |
| **Evaluation Metrics** | Retrieval precision, generation relevance, system latency |

**Visual Results:**

<img width="1079" height="409" alt="image" src="https://github.com/user-attachments/assets/0ddaa783-63e8-43ba-a24a-eb6ee0c0c577" />
<img width="912" height="341" alt="image" src="https://github.com/user-attachments/assets/2cbda8b3-02eb-4540-bd54-7e85f24ce3eb" />
<img width="1047" height="610" alt="image" src="https://github.com/user-attachments/assets/aaa6acd2-de5c-4d12-808b-c38f70ff31c6" />
<img width="747" height="215" alt="image" src="https://github.com/user-attachments/assets/f0f1a7d4-3a9b-4b8a-8661-ee3ecfe32bc6" />

---

## 💻 User Interface (UI)

**Production Interface Snapshots:**

<img width="1712" height="1122" alt="image" src="https://github.com/user-attachments/assets/d05bc056-c487-45e2-976e-491b80338141" />
<img width="1794" height="996" alt="image" src="https://github.com/user-attachments/assets/3809a526-abad-4b08-9fa0-476b26f2258a" />

---

## 📄 Further Details

A **detailed ablation report** is available in the repository:  
📘 [`RAG_Report.pdf`](./RAG_Report.pdf)

---

**Developed by:** *Team Manjiri — RAG_JioPay_BOT Project*  
**Repository:** [https://github.com/CODEX108/Manjiri-RAG-JioPay](https://github.com/CODEX108/Manjiri-RAG-JioPay)
