# RAG_Multilingual
📄 PDF RAG with Gemini AI

A multilingual PDF Question-Answering system powered by Google Gemini 2.5 models and LangChain.
Upload PDFs, index them into a vector store (FAISS), and ask questions in any language — including Arabic, Hindi, English, and more!

🚀 Features

📂 PDF Upload & Indexing – Extracts per-page text using pymupdf4llm.

🔍 Retrieval-Augmented Generation (RAG) – Uses FAISS + Gemini embeddings for context-aware answers.

🌍 Multilingual Support – Ask questions and upload PDFs in multiple languages (Arabic, French, English, etc.).

💬 Interactive Q&A – Chat with your documents using Gemini 2.5.

⚡ Fast and Lightweight – Built with Flask (backend) and a clean HTML/JS frontend.

🔐 Environment-First Config – Secure API key management with .env.

🛠️ Tech Stack

Backend: Flask + LangChain + Gemini (Generative AI + Embeddings)

Frontend: Vanilla HTML + CSS + JavaScript

Vector Store: FAISS (in local storage)

Document Processing: pymupdf4llm

📦 Installation
git clone https://github.com/yourusername/pdf-rag-gemini.git
cd pdf-rag-gemini
python -m venv .venv
source .venv/bin/activate  # (on Windows: .venv\Scripts\activate)
pip install -r requirements.txt

🔑 Environment Setup

Create a .env file in the root directory:

GEMINI_API_KEY=your_google_gemini_api_key

▶️ Run the App
python app.py


Backend will start on http://localhost:8050

Open http://localhost:8050 in your browser

💡 Example

Upload an Arabic Employee Handbook (PDF)

Ask in Arabic:

ما هي سياسة الإجازة في الشركة؟

Get context-aware answers sourced from the PDF.

⚖️ Safety & Limitations

❌ Model may hallucinate if relevant context is missing.

📚 Accuracy depends on PDF quality (OCR errors can affect results).

🌐 Requires Gemini API key from Google Cloud.

🔮 Future Improvements

🗂️ Support for DOCX, TXT in addition to PDF

🧠 Advanced reranking of retrieved chunks

💾 Cloud storage integration (GCS/S3)

🌍 UI translation for multilingual users

📌 Takeaway

This project shows how RAG + Gemini enables powerful multilingual document search & Q&A.
It’s lightweight, extendable, and ready for real-world use cases like enterprise document search, compliance, and knowledge assistants.

✨ Contributions welcome! Fork, improve, and PR 🚀
