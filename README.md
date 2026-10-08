# Gemini Document Embeddings with LangChain

## Overview

This project demonstrates how to generate **document embeddings using Google's Gemini Embedding model through LangChain**.

The implementation takes a list of text documents, splits them into manageable chunks using LangChain's `RecursiveCharacterTextSplitter`, and converts those chunks into numerical vector representations using the `GoogleGenerativeAIEmbeddings` class.

The project provides a simple foundation for understanding how document text can be transformed into embeddings for later use in applications such as semantic search, retrieval systems, and Retrieval-Augmented Generation (RAG).

---

## Objective

The main objectives of this project are to:

- Understand document embedding using Gemini.
- Integrate Gemini embeddings with LangChain.
- Split documents into manageable text chunks.
- Convert text chunks into numerical vector representations.
- Understand the basic document-to-embedding workflow.
- Learn how embedding models can prepare text for downstream retrieval applications.

---

## How It Works

The implementation follows a simple pipeline:

```text
Text Documents
      ↓
Recursive Character Text Splitter
      ↓
Document Chunks
      ↓
Gemini Embedding Model
      ↓
Embedding Vectors
      ↓
Vector Output
```

---

## 1. Loading Environment Variables

The project uses `python-dotenv` to load environment variables from a `.env` file.

```python
from dotenv import load_dotenv

load_dotenv()
```

The Gemini API key is retrieved securely from the environment:

```python
google_api_key=os.getenv("GOOGLE_API_KEY")
```

This avoids hardcoding the API key directly into the Python source code.

---

## 2. Setting Up the Gemini Embedding Model

The project uses LangChain's `GoogleGenerativeAIEmbeddings` class:

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
```

The `gemini-embedding-001` model converts text into numerical embedding vectors that represent the semantic characteristics of the input text.

These vectors can later be used by retrieval and similarity-based applications.

---

## 3. Preparing Documents

A small list of example documents is used:

```python
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]
```

These documents serve as the input to the text-processing and embedding pipeline.

---

## 4. Splitting Documents into Chunks

The project uses LangChain's `RecursiveCharacterTextSplitter`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
```

### Chunk Size

`chunk_size=1000` defines the maximum target size of each text chunk.

### Chunk Overlap

`chunk_overlap=100` allows neighboring chunks to share some text.

Chunking becomes especially important when working with larger documents because embedding models generally process text more effectively when documents are divided into manageable pieces.

---

## 5. Creating Document Objects

Instead of using `split_text()`, the implementation uses:

```python
doc_objects = text_splitter.create_documents(documents)
```

This is useful because the input is a **list of documents**.

The resulting objects contain the document text along with LangChain's document structure.

The actual text is then extracted using:

```python
chunks = [doc.page_content for doc in doc_objects]
```

This produces a list of text chunks that can be passed to the embedding model.

---

## 6. Generating Document Embeddings

The chunks are converted into vectors using:

```python
vector_list = embeddings.embed_documents(chunks)
```

Each input text chunk produces a corresponding numerical vector.

Conceptually:

```text
Document Chunk
      ↓
Gemini Embedding Model
      ↓
Numerical Vector
```

The project then prints a small preview of each generated vector:

```python
for i, vector in enumerate(vector_list):
    print(f"Vector {i+1} (Preview): {vector[:32]}...")
```

Only the first 32 values are displayed to keep the output readable.

---

## 7. Error Handling

The embedding operation is wrapped in a `try-except` block:

```python
try:
    vector_list = embeddings.embed_documents(chunks)

except Exception as e:
    print(f"Error: {e}")
```

This provides basic handling for issues that may occur during the embedding API call, such as configuration or API-related errors.

---

## End-to-End Workflow

The complete implementation can be summarized as:

### Step 1 — Load API Configuration

Load `GOOGLE_API_KEY` from the environment.

### Step 2 — Initialize Gemini Embeddings

Create the `GoogleGenerativeAIEmbeddings` object using `gemini-embedding-001`.

### Step 3 — Provide Documents

Pass a list of text documents to the application.

### Step 4 — Split Documents

Use `RecursiveCharacterTextSplitter` with:

- `chunk_size = 1000`
- `chunk_overlap = 100`

### Step 5 — Extract Text

Convert LangChain `Document` objects back into plain text chunks.

### Step 6 — Generate Embeddings

Send the chunks to Gemini using `embed_documents()`.

### Step 7 — Inspect Results

Print the number of chunks and a preview of each generated vector.

---

## Key Concepts

### Document Embeddings

Document embeddings are numerical representations of text.

Instead of representing a document only as words, an embedding model converts the text into a vector that captures useful semantic information.

For example:

```text
"Delhi is the capital of India"
              ↓
       Gemini Embedding Model
              ↓
   [0.012, -0.084, 0.231, ...]
```

These vectors can later be compared using similarity measures.

### Text Chunking

Large documents are commonly divided into smaller chunks before generating embeddings.

Chunking helps prepare documents for downstream applications such as:

- Semantic search
- Document retrieval
- Question answering
- RAG systems

### Vector Representation

The output of the embedding model is a numerical vector.

A collection of such vectors can later be stored in a vector database and used for similarity-based retrieval.

---

## Technologies Used

- **Python**
- **LangChain**
- **Google Gemini Embeddings**
- **langchain-google-genai**
- **langchain-text-splitters**
- **python-dotenv**

---

## Installation

Clone the repository and install the required packages:

```bash
pip install langchain-google-genai langchain-text-splitters python-dotenv
```

---

## API Key Configuration

Create a `.env` file in the project directory:

```env
GOOGLE_API_KEY=your_api_key_here
```

Make sure the `.env` file is included in `.gitignore` so that your API key is not committed to GitHub.

Example:

```gitignore
.env
```

---

## Running the Project

Run the Python file:

```bash
python embedding_gemini_docs.py
```

The program will display:

1. The total number of text chunks.
2. A preview of the generated embedding vector for each chunk.
3. Any error encountered during the embedding process.

---

## Learning Outcomes

After working through this project, you should understand:

- What document embeddings are.
- How Gemini can generate embeddings from text.
- How LangChain integrates with Gemini embedding models.
- Why documents are split into chunks.
- How `RecursiveCharacterTextSplitter` works at a basic level.
- The difference between LangChain `Document` objects and plain text.
- How `embed_documents()` generates vectors for multiple text chunks.
- How embeddings can serve as the foundation for semantic retrieval systems.

---

## Future Improvements

This basic embedding pipeline can be extended into a complete retrieval system by adding:

- Vector databases such as FAISS or Chroma.
- Similarity search.
- Query embeddings.
- Top-k document retrieval.
- Metadata filtering.
- Document loaders for PDFs, websites, and other sources.
- RAG pipelines with an LLM.
- Retrieval evaluation and benchmarking.
- Hybrid keyword + semantic search.
- Reranking of retrieved documents.

These additions would transform the current document-embedding demonstration into a more complete **semantic retrieval or RAG application**.

---

## Applications

Gemini document embeddings can serve as a building block for applications such as:

- Semantic search
- Knowledge-base retrieval
- Document question answering
- RAG applications
- Recommendation systems
- Similarity-based document matching
- Enterprise knowledge retrieval

---

## Conclusion

This project demonstrates the basic workflow for generating **document embeddings using Gemini and LangChain**.

It starts with a list of text documents, splits them using `RecursiveCharacterTextSplitter`, extracts the resulting text chunks, and generates numerical vector representations using Google's `gemini-embedding-001` embedding model.

Although the implementation is intentionally simple, it provides an important foundation for understanding how modern **semantic retrieval and RAG systems** process documents before retrieval and generation.
