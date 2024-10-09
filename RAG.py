import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAG:
    def __init__(self, pdf_path, model_name="all-MiniLM-L6-v2"):
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.text_chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.embedding_model = SentenceTransformer(model_name)
        self._load_data()

    # Step 1: Extract Text from PDF
    def _extract_text_from_pdf(self):
        text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    # Step 2: Split Text into Chunks
    def _split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
        return chunks

    # Step 3: Generate Embeddings
    def _get_embeddings(self, text_chunks):
        embeddings = self.embedding_model.encode(text_chunks)
        return embeddings

    # Step 4: Create FAISS Index
    def _create_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    # Load and process data
    def _load_data(self):
        # Extract and process the PDF text
        text = self._extract_text_from_pdf()
        self.text_chunks = self._split_text(text)
        self.embeddings = self._get_embeddings(self.text_chunks)
        self.faiss_index = self._create_faiss_index(np.array(self.embeddings))

    # Step 5: Search the FAISS Index
    def search_index(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        results = [self.text_chunks[i] for i in indices[0]]
        return results
