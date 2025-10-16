"""
Vector Store Module
Manages ChromaDB for storing and retrieving document chunks with local embeddings
"""
import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import hashlib
from dotenv import load_dotenv

load_dotenv()


class LocalEmbeddingFunction:
    """Local embedding function using sentence-transformers with robust type handling"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize local embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Check GPU availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"🔧 Loading embedding model '{model_name}' on {device.upper()}...")
            self.model = SentenceTransformer(model_name, device=device)
            self.device = device
            print(f"✅ Embedding model loaded on {device.upper()}")
            
        except Exception as e:
            print(f"⚠️ Error loading sentence-transformers: {e}")
            print("📦 Falling back to ChromaDB default embeddings")
            self.model = None
    
    def embed_query(self, input: str) -> List[float]:
        """
        Generate embedding for a single query text.
        ChromaDB calls this when querying the database.
        Returns: List[float] - a single embedding vector
        """
        # Ensure input is a string (not a list or other type)
        if isinstance(input, list):
            print(f"⚠️ Warning: embed_query received a list, taking first element")
            input_str = str(input[0]) if input else ""
        else:
            input_str = str(input)
        
        print(f"🔍 Embedding query: '{input_str[:50]}...'")
        
        # Use fallback if model not loaded
        if self.model is None:
            fallback = self._fallback_embeddings([input_str])
            result = fallback[0]
            # Ensure it's a true Python list of floats
            result = [float(x) for x in result]
            return result
        
        try:
            # Generate embedding using local model (single text)
            import numpy as np
            embedding = self.model.encode(input_str, convert_to_numpy=True, show_progress_bar=False)
            
            # CRITICAL: Ensure we have a 1D numpy array
            if len(embedding.shape) > 1:
                print(f"⚠️ Warning: embedding has shape {embedding.shape}, flattening")
                embedding = embedding.flatten()
            
            # Convert to Python list of native Python floats (not numpy floats)
            emb_list = embedding.tolist()
            
            # Double-check: ensure it's a list and convert all to native Python float
            if not isinstance(emb_list, list):
                emb_list = list(emb_list)
            
            # Convert each element to native Python float (not np.float32/np.float64)
            result = []
            for x in emb_list:
                if isinstance(x, (np.floating, np.integer)):
                    result.append(float(x))  # Convert numpy types to Python float
                else:
                    result.append(float(x))
            
            # Final validation
            assert isinstance(result, list), f"Result is not a list: {type(result)}"
            assert len(result) > 0, "Result is empty"
            assert all(isinstance(x, float) and not isinstance(x, np.floating) for x in result), "Not all elements are Python floats"
            
            # Extra debug: check a few elements
            print(f"✅ Query embedding: {len(result)} dims")
            print(f"   Type: {type(result)}")
            print(f"   First 3 elements: {result[:3]}")
            print(f"   Element types: {[type(x).__name__ for x in result[:3]]}")
            print(f"   Is pure list: {result.__class__.__name__}")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error generating query embedding: {e}")
            import traceback
            traceback.print_exc()
            fallback = self._fallback_embeddings([input_str])
            result = [float(x) for x in fallback[0]]
            return result
    
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        """
        Generate embeddings for a list of texts.
        ChromaDB calls this method with Documents (List[str]) and expects Embeddings (List[List[float]])
        """
        # Ensure input is a list
        if not isinstance(input, list):
            input = [str(input)]
        else:
            # Ensure all items are strings
            input = [str(item) for item in input]
        
        print(f"🔍 Embedding {len(input)} document(s)...")
        
        # Use fallback if model not loaded
        if self.model is None:
            return self._fallback_embeddings(input)
        
        try:
            # Generate embeddings using local model
            import numpy as np
            embeddings = self.model.encode(input, convert_to_numpy=True, show_progress_bar=False)
            
            print(f"🔍 Batch embeddings shape: {embeddings.shape}")
            
            # Convert numpy arrays to Python lists
            result: List[List[float]] = []
            
            for idx, emb in enumerate(embeddings):
                try:
                    # Convert to Python list
                    emb_list = emb.tolist()
                    
                    # Convert all numpy types to native Python float
                    emb_floats = []
                    for x in emb_list:
                        if isinstance(x, (np.floating, np.integer)):
                            emb_floats.append(float(x))
                        else:
                            emb_floats.append(float(x))
                    
                    # Validate
                    assert isinstance(emb_floats, list), f"Not a list: {type(emb_floats)}"
                    assert all(isinstance(x, float) and not isinstance(x, np.floating) for x in emb_floats), f"Not all Python floats"
                    
                    result.append(emb_floats)
                    
                except Exception as e:
                    print(f"⚠️ Error converting embedding {idx}: {e}")
                    # Use fallback for this specific embedding
                    fallback = self._fallback_embeddings([input[idx]])
                    result.append([float(x) for x in fallback[0]])
            
            print(f"✅ Generated {len(result)} embeddings, each with {len(result[0]) if result else 0} dimensions")
            return result
            
        except Exception as e:
            print(f"⚠️ Embedding error: {e}, using fallback for all texts")
            import traceback
            traceback.print_exc()
            return self._fallback_embeddings(input)
    
    def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fallback embeddings using ChromaDB default"""
        print(f"⚠️ Using fallback embeddings for {len(texts)} text(s)")
        try:
            from chromadb.utils import embedding_functions
            default_ef = embedding_functions.DefaultEmbeddingFunction()
            embeddings = default_ef(texts)
            
            # Ensure embeddings are lists of lists of floats
            result: List[List[float]] = []
            for idx, emb in enumerate(embeddings):
                try:
                    if hasattr(emb, 'tolist'):
                        emb_list = emb.tolist()
                    elif isinstance(emb, list):
                        emb_list = emb
                    elif hasattr(emb, '__iter__'):
                        emb_list = list(emb)
                    else:
                        # Emergency fallback: create zero embedding
                        emb_list = [0.0] * 384
                    
                    # Ensure all elements are floats
                    emb_floats = [float(x) for x in emb_list]
                    result.append(emb_floats)
                    
                except Exception as e:
                    print(f"⚠️ Error in fallback {idx}: {e}")
                    # Emergency: zero embedding
                    result.append([0.0] * 384)
            
            return result
            
        except Exception as e:
            print(f"⚠️ Critical error in fallback: {e}")
            # Ultimate fallback: return zero embeddings for all texts
            return [[0.0] * 384 for _ in texts]


class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "pdf_documents"):
        """Initialize ChromaDB vector store with local embeddings"""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create local embedding function
        self.embedding_function = LocalEmbeddingFunction()
        
        # Initialize image metadata storage
        self.image_metadata = {}
        
        # Create client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection with custom embedding function
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print("✅ Using existing collection with local embeddings")
        except Exception as e:
            # If collection exists with different embedding, delete and recreate
            try:
                self.client.delete_collection(name=collection_name)
                print("🔄 Deleted old collection (different embedding function)")
            except:
                pass
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Created new collection with local embeddings")
    
    def chunk_text(self, pages_content: List[Dict], described_images: List[Dict] = None, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """
        Split pages into chunks with metadata
        Now includes image descriptions merged into page text for better context!
        """
        chunks = []
        
        # Create image lookup by page
        image_by_page = {}
        if described_images:
            for img_data in described_images:
                page_num = img_data.get("page", 0)
                if page_num not in image_by_page:
                    image_by_page[page_num] = []
                image_by_page[page_num].append(img_data)
        
        for page_data in pages_content:
            page_num = page_data.get("page", 1)
            content = page_data.get("content", "")
            
            # 🔥 MERGE IMAGE DESCRIPTIONS into page text
            if page_num in image_by_page:
                page_images = image_by_page[page_num]
                content += "\n\n[IMAGES ON THIS PAGE]:\n"
                for idx, img in enumerate(page_images, 1):
                    img_desc = img.get("description", "No description")
                    img_filename = img.get("filename", "unknown")
                    content += f"Image {idx} ({img_filename}): {img_desc}\n"
                print(f"📸 Page {page_num}: Merged {len(page_images)} image description(s) into text")
            
            # Split content into smaller chunks
            words = content.split()
            current_chunk = []
            current_size = 0
            
            for word in words:
                current_chunk.append(word)
                current_size += len(word) + 1  # +1 for space
                
                if current_size >= chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "type": "text"
                    })
                    
                    # Keep overlap words for next chunk
                    overlap_words = int(len(current_chunk) * (overlap / chunk_size))
                    current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                    current_size = sum(len(w) + 1 for w in current_chunk)
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "type": "text"
                })
        
        return chunks
    
    def _generate_chunk_id(self, text: str, page: int, index: int) -> str:
        """Generate unique ID for a chunk"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"page_{page}_chunk_{index}_{content_hash}"
    
    def add_documents(self, processed_data: Dict, described_images: List[Dict]):
        """
        Add processed PDF data to vector store.
        Note: pages_content already includes image descriptions merged into the text.
        """
        pages_content = processed_data.get("pages", [])
        pdf_name = processed_data.get("pdf_name", "unknown")
        
        # Chunk text content (now includes image descriptions)
        text_chunks = self.chunk_text(pages_content)
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        # Add text chunks (which now include image descriptions contextually)
        for idx, chunk in enumerate(text_chunks):
            chunk_id = self._generate_chunk_id(chunk["text"], chunk["page"], idx)
            
            documents.append(chunk["text"])
            metadatas.append({
                "page": chunk["page"],
                "type": "text",
                "pdf_name": pdf_name,
                "chunk_index": idx
            })
            ids.append(chunk_id)
        
        # Store image metadata (for tracking which images exist on which pages)
        # Note: We don't add descriptions as separate searchable text since they're 
        # already merged into page content above
        image_metadata = {}
        for idx, img_data in enumerate(described_images):
            page = img_data["page"]
            if page not in image_metadata:
                image_metadata[page] = []
            image_metadata[page].append({
                "path": img_data["path"],
                "filename": img_data["filename"],
                "description": img_data["description"]
            })
        
        # Store image metadata in collection metadata (for later retrieval)
        self.image_metadata = image_metadata
        
        # Add to ChromaDB
        if documents:
            print(f"📝 Adding {len(documents)} text chunks to vector store (enriched with {len(described_images)} image descriptions)")
            
            # Add in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                print(f"  ✅ Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            print(f"✅ Successfully added all documents to vector store")
    
    def query(self, query_text: str, n_results: int = 5) -> Dict:
        """
        Query the vector store
        """
        print(f"🔍 Querying vector store: '{query_text[:50]}...'")
        
        # Generate embedding manually to ensure proper format
        
        query_embedding = self.embedding_function.embed_query(query_text)

        # Safety: handle accidental float or empty embedding
        if isinstance(query_embedding, float):
            print("⚠️ embed_query returned a float instead of list, wrapping it")
            query_embedding = [query_embedding]
        elif not isinstance(query_embedding, (list, tuple)):
            print(f"⚠️ Unexpected type for query_embedding: {type(query_embedding)}, converting to list")
            query_embedding = [float(query_embedding)]

        # Ensure all elements are floats
        query_embedding = [float(x) for x in query_embedding if isinstance(x, (int, float))]

        # Double wrap for ChromaDB format (List[List[float]])
        query_embeddings = [query_embedding]

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )


        
        # Format results
        formatted_results = {
            "documents": [],
            "metadatas": [],
            "distances": results["distances"][0] if results["distances"] else []
        }
        
        if results["documents"]:
            formatted_results["documents"] = results["documents"][0]
        
        if results["metadatas"]:
            formatted_results["metadatas"] = results["metadatas"][0]
        
        print(f"✅ Found {len(formatted_results['documents'])} results")
        return formatted_results
    
    def get_images_for_page(self, page_num: int) -> List[Dict]:
        """Get all images for a specific page"""
        return self.image_metadata.get(page_num, [])
    
    def clear_collection(self):
        """Clear all documents from collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Collection cleared")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name
            }
        except:
            return {"total_chunks": 0, "collection_name": self.collection_name}
    
    def is_pdf_processed(self, pdf_name: str) -> bool:
        """Check if a PDF has already been processed"""
        try:
            results = self.collection.get(
                where={"pdf_name": pdf_name},
                limit=1
            )
            return len(results['ids']) > 0
        except Exception as e:
            print(f"Error checking if PDF processed: {e}")
            return False
    
    def get_processed_pdfs(self) -> List[str]:
        """Get list of processed PDF names"""
        try:
            # Get all unique PDF names from metadata
            all_data = self.collection.get()
            if all_data and all_data['metadatas']:
                pdf_names = set()
                for meta in all_data['metadatas']:
                    if 'pdf_name' in meta:
                        pdf_names.add(meta['pdf_name'])
                return sorted(list(pdf_names))
            return []
        except Exception as e:
            print(f"Error getting processed PDFs: {e}")
            return []
