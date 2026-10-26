import os
import shutil
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # langchain-community sunset notice
from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DOCUMENT_PATH = "Document.html"

DB_DIR = "./document_db"
DONE_MARKER = os.path.join(DB_DIR, "BUILD_COMPLETE")
CHUNK_CHARS = 1500  # Chunk size
BATCH = 64
TOP_K = 6

QUESTION = ("What percentage of Ares Capital's new investment commitments in the "
            "fourth quarter of 2025 were first lien senior secured loans?")


def load_chunks(path):
    soup = BeautifulSoup(open(path, "rb"), "html.parser")
    blocks = [el.get_text(" ", strip=True) for el in soup.find_all(["div", "tr"]) if not el.find(["div", "tr"])]
    blocks = [b for b in blocks if b]

    chunks, current, first = [], [], 0
    for i, block in enumerate(blocks):
        current.append(block)
        if sum(map(len, current)) >= CHUNK_CHARS or i == len(blocks) - 1:
            chunks.append(Document(page_content="\n".join(current),
                                   metadata={"first_block": first, "last_block": i}))
            current, first = [block], i
    return chunks


embeddings = FastEmbedEmbeddings()

if os.path.exists(DONE_MARKER):
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    shutil.rmtree(DB_DIR, ignore_errors=True)
    chunks = load_chunks(DOCUMENT_PATH)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    for i in tqdm(range(0, len(chunks), BATCH), desc="Embedding"):
        db.add_documents(chunks[i:i + BATCH])
    open(DONE_MARKER, "w").close()

docs = db.similarity_search(QUESTION, k=TOP_K)
context = "\n\n".join(f"[Source {i+1}]\n{d.page_content}" for i, d in enumerate(docs))

prompt = ChatPromptTemplate.from_template(
    "Answer using only the context below. Cite sources like [Source 2] and quote the supporting text. "
    "If the answer is not in the context, say so.\n\nContext:\n{context}\n\nQuestion: {question}"
)
chain = prompt | OllamaLLM(model="gemma4:12b-it-qat", num_ctx=8192) | StrOutputParser()
print(f"QUESTION: {QUESTION}\n\n--- SOURCES ---")
for i, doc in enumerate(docs):
    m = doc.metadata
    print(f"\n[Source {i+1} | blocks {m.get('first_block')}-{m.get('last_block')}]")
    print(doc.page_content.strip()[:250] + "...\n")

print("--- ANSWER ---")
print(chain.invoke({"context": context, "question": QUESTION}))
