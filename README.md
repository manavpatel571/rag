# Advanced PDF RAG System with Image Descriptions

An intelligent PDF question-answering system powered by AI that understands both text and images in your documents.

## Features

- **PDF Processing**: Converts PDFs to structured markdown using Docling
- **Image Understanding**: Automatically extracts and describes images using Qwen VL model
- **Smart Q&A**: Ask questions and get accurate answers with proper citations
- **Image Context**: Retrieves and displays relevant images alongside answers
- **Citation System**: Shows page numbers, text snippets, and source references
- **Modern UI**: Beautiful Streamlit interface with chat-based interaction

## Technologies

- **Docling**: PDF to Markdown conversion
- **Qwen2.5-VL-7B-Instruct**: Advanced AI-powered image understanding via HuggingFace
- **Qwen2.5-7B-Instruct**: Natural language chat responses via HuggingFace
- **all-MiniLM-L6-v2**: Local embedding model for semantic search (runs on device)
- **ChromaDB**: Vector database for storing embeddings
- **Streamlit**: Interactive web interface
- **HuggingFace API**: API for chat and vision models

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Streamlit for the UI
- OpenAI Python client (for HuggingFace API)
- ChromaDB for vector storage
- Docling for PDF processing
- And other dependencies

### 2. Configure API Key

Create a `.env` file in the project root:

```env
# HuggingFace API Key (REQUIRED)
HF_API_KEY=your_huggingface_api_key_here
```

**Getting Your API Key:**
1. Visit [HuggingFace](https://huggingface.co/)
2. Sign up or log in
3. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
4. Click "New token"
5. Give it a name and select "Read" permission
6. Copy the token and paste it in `.env`

**Note**: All models (chat, image, embeddings) run via HuggingFace API!

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

1. **Upload PDF**: Click "Choose a PDF file" in the sidebar
2. **Process**: Click "Process PDF" to extract text and images
3. **Ask Questions**: Type your questions in the chat input
4. **Get Answers**: Receive AI-powered responses with citations and relevant images

## Project Structure

```
nlp_rag/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # API keys (create this)
├── README.md                   # Documentation
├── utils/                      # Core modules
│   ├── __init__.py
│   ├── pdf_processor.py        # PDF text & image extraction
│   ├── image_describer.py      # Image description with Qwen VL
│   ├── vector_store.py         # ChromaDB vector storage
│   └── rag_engine.py           # RAG logic with Qwen2.5-7B
├── extracted_images/           # Extracted PDF images (auto-created)
└── chroma_db/                  # Vector database (auto-created)
```

## How It Works

1. **PDF Upload & Processing**
   - Extracts text and converts to markdown
   - Extracts all images from the PDF
   - Saves images with page references

2. **Image Description**
   - Sends images to Qwen VL model via HuggingFace API
   - Generates detailed descriptions
   - Links descriptions to page numbers

3. **Vector Storage**
   - Chunks text into manageable segments
   - Combines text with image descriptions
   - Creates embeddings and stores in ChromaDB

4. **Question Answering**
   - Retrieves relevant chunks based on query
   - Formats context with text and image info
   - Generates answer using Qwen2.5-7B-Instruct
   - Returns answer with citations and images

## Example Questions

- "What are the main topics covered in this document?"
- "Summarize the findings on page 5"
- "What do the charts and images show?"
- "Explain the methodology described in the paper"

## Notes

- First-time processing may take a few minutes depending on PDF size
- Image descriptions and chat responses require HuggingFace API credits
- HuggingFace API has rate limits on the free tier
- Extracted images are stored locally in `extracted_images/`
- Embeddings run locally (no API costs)

## Troubleshooting

**PDF Processing Fails:**
- Ensure the PDF is not password-protected
- Check if the PDF has selectable text (not just scanned images)

**Image Descriptions Not Working:**
- Verify your HuggingFace API key is valid
- Check if you have API credits/quota remaining

**Chat Responses Error:**
- Verify your HuggingFace API key is correct
- Check HuggingFace API quota limits
- Ensure you have access to Qwen models on HuggingFace

## License

MIT License - feel free to use and modify as needed.

