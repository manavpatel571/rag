"""
RAG Engine Module
Handles query processing and response generation with Qwen2.5-7B-Instruct
"""
import os
from typing import List, Dict, Tuple
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()


class RAGEngine:
    def __init__(self, vector_store):
        """Initialize RAG engine with Qwen2.5-7B-Instruct via HuggingFace API"""
        self.vector_store = vector_store
        
        # Configure HuggingFace InferenceClient
        api_key = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
        
        if not api_key:
            print("⚠️ Warning: HF_API_KEY not found in environment")
            self.client = None
        else:
            self.client = InferenceClient(
                model="Qwen/Qwen2.5-7B-Instruct",
                token=api_key
            )
            print("✅ Qwen2.5-7B-Instruct initialized for chat responses")
    
    def format_context(self, retrieved_chunks: Dict) -> Tuple[str, List[Dict]]:
        """
        Format retrieved chunks into context for LLM
        Returns: (context_text, citations_data)
        Note: Retrieved chunks already include image descriptions merged into the text.
        """
        context_parts = []
        citations = []
        
        documents = retrieved_chunks.get("documents", [])
        metadatas = retrieved_chunks.get("metadatas", [])
        
        for idx, (doc, metadata) in enumerate(zip(documents, metadatas)):
            page = metadata.get("page", "Unknown")
            
            # The document text already includes image descriptions
            context_entry = f"[Source {idx + 1} - Page {page}]\n{doc}"
            context_parts.append(context_entry)
            
            # Get images for this page (if any)
            page_images = self.vector_store.get_images_for_page(page)
            image_paths = [img["path"] for img in page_images] if page_images else []
            
            # Prepare citation data
            citations.append({
                "source_id": idx + 1,
                "page": page,
                "text_snippet": doc[:150] + "..." if len(doc) > 150 else doc,
                "image_paths": image_paths,
                "has_images": len(image_paths) > 0
            })
        
        context_text = "\n\n".join(context_parts)
        return context_text, citations
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using Qwen3-4B with context
        """
        if self.client is None:
            return "Error: HF_API_KEY not configured"
        
        system_prompt = """You are a helpful AI assistant answering questions based on the provided document context.

Instructions:
- Answer the question based ONLY on the provided context
- If the answer is not in the context, say so clearly
- Be concise and accurate
- Reference specific page numbers when relevant
- If images are mentioned in the context, acknowledge them in your answer"""
        
        user_prompt = f"""Context from the document:
{context}

User Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1024,
                temperature=0.3
            )
            
            # Extract content from response
            return response.choices[0].message["content"]
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Error in generate_response:")
            print(error_details)
            return f"Error generating response: {str(e)}"
    
    def query_and_respond(self, query: str, n_results: int = 5) -> Dict:
        """
        Main method: retrieve context and generate response
        Returns: {
            "answer": str,
            "citations": List[Dict],
            "model_used": str
        }
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.vector_store.query(query, n_results=n_results)
        
        # Format context and get citations
        context, citations = self.format_context(retrieved_chunks)
        
        # Generate response
        answer = self.generate_response(query, context)
        
        return {
            "answer": answer,
            "citations": citations,
            "model_used": "Qwen2.5-7B-Instruct",
            "query": query
        }
    
    def chat_with_history(self, query: str, chat_history: List[Dict], n_results: int = 5) -> Dict:
        """
        Generate response considering chat history
        """
        if self.client is None:
            return {
                "answer": "Error: HF_API_KEY not configured",
                "citations": [],
                "model_used": "Qwen2.5-7B-Instruct",
                "query": query
            }
        
        # Retrieve relevant chunks
        retrieved_chunks = self.vector_store.query(query, n_results=n_results)
        
        # Format context and get citations
        context, citations = self.format_context(retrieved_chunks)
        
        # Build messages with history
        messages = [
            {
                "role": "system",
                "content": """You are a helpful AI assistant answering questions based on the provided document context.

Instructions:
- Answer the question based on the provided context and conversation history
- If the answer is not in the context, say so clearly
- Be concise and accurate
- Reference specific page numbers when relevant
- If images are mentioned in the context, acknowledge them in your answer"""
            }
        ]
        
        # Add conversation history (last 3 exchanges)
        for msg in chat_history[-6:]:  # Last 3 Q&A pairs
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"]:
                messages.append({"role": role, "content": content})
        
        # Add current context and question
        user_message = f"""Context from the document:
{context}

User Question: {query}"""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                max_tokens=1024,
                temperature=0.3
            )
            
            # Extract content from response
            answer = response.choices[0].message["content"]
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Error in chat_with_history:")
            print(error_details)
            answer = f"Error generating response: {str(e)}"
        
        return {
            "answer": answer,
            "citations": citations,
            "model_used": "Qwen2.5-7B-Instruct",
            "query": query
        }

