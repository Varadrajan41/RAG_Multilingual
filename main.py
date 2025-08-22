



import json
import os
import uuid
import pathlib
from typing import List

from flask import Flask, jsonify, request, send_file, send_from_directory

# LangChain + Gemini
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# PDF to markdown chunks
import pymupdf4llm

# asyncio loop helper
import asyncio

from dotenv import load_dotenv

# ---------- Setup ----------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Models
GENERATION_MODEL = "gemini-2.5-flash"   # or "gemini-2.5-pro" if you want slower but smarter
EMBED_MODEL = "text-embedding-004"

# Paths
DATA_DIR = pathlib.Path("./data")
INDEX_DIR = DATA_DIR / "faiss_index"
UPLOAD_DIR = DATA_DIR / "uploads"
for p in (DATA_DIR, INDEX_DIR, UPLOAD_DIR):
    p.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# --------- Helpers ---------
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

def get_embedder():
    ensure_event_loop()
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

def load_index(embedder: GoogleGenerativeAIEmbeddings) -> FAISS | None:
    faiss_path = INDEX_DIR / "index.faiss"
    pkl_path = INDEX_DIR / "index.pkl"
    if faiss_path.exists() and pkl_path.exists():
        return FAISS.load_local(
            str(INDEX_DIR), embedder, allow_dangerous_deserialization=True
        )
    return None

def save_index(index: FAISS):
    index.save_local(str(INDEX_DIR))

def pdf_to_docs(pdf_path: str, title: str) -> List[Document]:
    data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    docs: List[Document] = []
    for i, page in enumerate(data):
        text = page.get("markdown") or page.get("text") or ""
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": pathlib.Path(pdf_path).name,
                    "title": title,
                    "page": i + 1,
                },
            )
        )
    return docs

# --------- Routes ---------
@app.route("/")
def index():
    return send_file("web/index.html")

@app.route("/api/generate", methods=["POST"])
def generate_api():
    if os.environ.get("GOOGLE_API_KEY", "TODO") in ["TODO", "", None]:
        return jsonify({"error": "Add your Gemini API key in app.py or set GOOGLE_API_KEY env var."})
    try:
        req_body = request.get_json()
        content = req_body.get("contents")
        model_name = req_body.get("model", GENERATION_MODEL)
        model = ChatGoogleGenerativeAI(model=model_name)
        message = HumanMessage(content=content)
        response = model.stream([message])

        def stream():
            for chunk in response:
                yield "data: %s\n\n" % json.dumps({"text": chunk.content})
        return stream(), {"Content-Type": "text/event-stream"}
    except Exception as e:
        return jsonify({"error": str(e)})

# ---------- RAG endpoints ----------
@app.route("/api/ingest_pdf", methods=["POST"])
def ingest_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported"}), 400

        # Save upload
        uid = uuid.uuid4().hex[:8]
        save_path = UPLOAD_DIR / f"{uid}_{file.filename}"
        file.save(str(save_path))

        # Convert to chunks
        title = request.form.get("title", file.filename)
        docs = pdf_to_docs(str(save_path), title)

        # Build or load index
        embedder = get_embedder()
        index = load_index(embedder)
        if index is None:
            index = FAISS.from_documents(docs, embedder)
        else:
            index.add_documents(docs)

        save_index(index)

        return jsonify({"ok": True, "file": file.filename, "chunks": len(docs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        body = request.get_json()
        question = (body.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Missing 'question'"}), 400

        k = int(body.get("k", 4))
        model_name = body.get("model", GENERATION_MODEL)

        embedder = get_embedder()
        index = load_index(embedder)
        if index is None:
            return jsonify({"error": "Index is empty. Ingest a PDF first."}), 400

        retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": k})
        contexts = retriever.get_relevant_documents(question)

        context_snippets = []
        for d in contexts:
            meta = d.metadata or {}
            tag = f"[{meta.get('title','')}, p.{meta.get('page','?')}]"
            context_snippets.append(f"{tag}\n{d.page_content}")
        context_block = "\n\n---\n\n".join(context_snippets) if context_snippets else "No context found."

        # System prompt with multilingual support
        system_hint = (
            "You are a helpful assistant. Support multiple languages (English, Spanish, Arabic). "
            "Answer ONLY using the provided context. If the answer is not in the context, say "
            "\"I can’t find that in the provided PDF context.\""
        )
        user_prompt = f"{system_hint}\n\n# Question\n{question}\n\n# Context\n{context_block}\n"

        llm = ChatGoogleGenerativeAI(model=model_name)
        msg = HumanMessage(content=user_prompt)
        result = llm.invoke([msg])
        answer_main = result.content

        # Optional: also return English translation if question is NOT English
        trans_prompt = f"Translate the following answer into English:\n\n{answer_main}"
        trans_msg = HumanMessage(content=trans_prompt)
        trans_result = llm.invoke([trans_msg])
        answer_en = trans_result.content.strip()

        return jsonify({
            "answer": answer_main,
            "answer_english": answer_en if answer_en != answer_main else None,
            "sources": [
                {
                    "source": (d.metadata or {}).get("source"),
                    "title": (d.metadata or {}).get("title"),
                    "page": (d.metadata or {}).get("page")
                }
                for d in contexts
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 8050)))
