# LabGPT RAG Document Processor

A web-based application for processing laboratory documents to create embeddings and search indices for RAG (Retrieval-Augmented Generation) systems.

## Features

- **Web Interface**: Modern, responsive UI following the LabGPT design patterns
- **Document Processing**: Supports PDF, TXT, MD, and other text-based formats
- **Two-Stage Pipeline**: 
  1. Data Ingestion - Extract text and create chunks
  2. Index Building - Create FAISS search index
- **Real-time Progress**: Live monitoring of processing status
- **Multiple Output Formats**: Generates chunks, embeddings, and search index

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python rag_app.py
```

3. Open your browser and navigate to:
```
http://localhost:5001
```

## Usage

### 1. Start Processing

1. Enter the directory path containing your lab documents
2. **Override Option**: If existing RAG output files are detected, you can choose to:
   - **Keep existing files**: Processing will be blocked to prevent accidental overwriting
   - **Override existing files**: Existing outputs will be deleted and new ones generated
3. Click "Start Processing" to begin the pipeline
4. Monitor progress in real-time

### 2. Processing Stages

**Stage 0: Cleanup (Optional)**
- Removes existing output files if override option is enabled
- Ensures clean processing environment
- Logs which files are being removed

**Stage 1: Data Ingestion**
- Reads documents from the specified directory
- Extracts text using PyMuPDF for PDFs and standard readers for text files
- Splits text into overlapping chunks (2048 characters with 512 overlap)
- Creates embeddings using sentence-transformers/all-mpnet-base-v2

**Stage 2: Index Building**
- Builds FAISS HNSWFlat index for efficient similarity search
- Optimizes for both CPU and GPU (if available)
- Saves the final index to disk

### 3. Output Files

The processing generates three main files:

- `chunks.npy` - Text chunks from your documents
- `embeddings.npy` - Vector embeddings for semantic search  
- `faiss_index.bin` - Optimized search index

## Using the Generated Files

### Python Integration

```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the generated files
chunks = np.load("chunks.npy", allow_pickle=True)
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_index.bin")

# Initialize the embedding model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Perform a search
query = "your search query"
query_embedding = model.encode([query])
distances, indices = index.search(query_embedding, k=5)

# Get relevant chunks
relevant_chunks = [chunks[i] for i in indices[0]]
```

## Configuration

The application uses these default settings:

- **Chunk Size**: 2048 characters (~512 tokens)
- **Chunk Overlap**: 512 characters (~128 tokens)
- **Embedding Model**: sentence-transformers/all-mpnet-base-v2
- **Index Type**: FAISS HNSWFlat
- **Port**: 5001

## Supported File Formats

- PDF files (.pdf)
- Text files (.txt)
- Markdown files (.md)
- Other text-based formats

## API Endpoints

- `GET /` - Main application page
- `POST /` - Start processing with directory path and override option
- `GET /processing` - Processing status page
- `GET /api/status` - JSON status endpoint
- `GET /api/existing-files` - Check for existing output files
- `GET /results` - Results and output files page
- `GET /reset` - Reset processing status

## Technical Details

### Data Ingestion
- Uses PyMuPDF for PDF text extraction
- Employs LangChain's RecursiveCharacterTextSplitter for optimal chunking
- Creates dense embeddings using state-of-the-art sentence transformers

### Index Building
- Implements FAISS HNSWFlat algorithm for efficient similarity search
- Supports both CPU and GPU processing
- Optimized for high-dimensional vector search

### Performance
- Processes documents in batches with progress tracking
- Real-time log streaming
- Efficient memory usage for large document collections

## Troubleshooting

### Common Issues

1. **Directory not found**: Ensure the path exists and contains supported files
2. **Permission errors**: Check read permissions for the document directory
3. **Memory issues**: Large document collections may require more RAM
4. **CUDA errors**: GPU processing falls back to CPU automatically

### Log Monitoring

The application provides real-time logs showing:
- Document processing progress
- Embedding generation status
- Index building progress
- Error messages and warnings

## Development

### File Structure
```
RAG/
├── rag_app.py              # Main Flask application
├── data_ingestion.py       # Document processing script
├── index_builder.py        # FAISS index creation script
├── templates/              # HTML templates
│   ├── rag_index.html      # Main page
│   ├── rag_processing.html # Processing status
│   └── rag_results.html    # Results page
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the existing code style
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the LabGPT suite of tools for scientific document processing. 