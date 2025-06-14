from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
import logging

class DocumentReranker:
    """Document reranker using cross-encoder models for improved relevance scoring"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker with a cross-encoder model
        
        Args:
            model_name: Name of the cross-encoder model to use
                       Default is a multilingual model that works well with Vietnamese
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the cross-encoder model"""
        try:
            print(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            print("Reranker model loaded successfully")
        except Exception as e:
            print(f"Error loading reranker model: {e}")
            # Fallback to a lighter model
            try:
                fallback_model = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
                print(f"Trying fallback model: {fallback_model}")
                self.model = CrossEncoder(fallback_model)
                self.model_name = fallback_model
                print("Fallback reranker model loaded successfully")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                self.model = None
    
    def rerank_documents(self, 
                        query: str, 
                        documents: List[Dict[str, Any]], 
                        top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return (None for all)
            
        Returns:
            List of reranked documents with updated scores
        """
        if not self.model or not documents:
            return documents
        
        try:
            # Prepare query-document pairs for the cross-encoder
            pairs = []
            valid_docs = []
            
            for doc in documents:
                content = doc.get('content', '')
                title = doc.get('title', '')
                
                # Combine title and content for better matching
                doc_text = f"{title}. {content}" if title else content
                
                # Truncate very long documents to avoid model limits
                max_length = 512
                if len(doc_text) > max_length:
                    doc_text = doc_text[:max_length] + "..."
                
                pairs.append([query, doc_text])
                valid_docs.append(doc)
            
            if not pairs:
                return documents
            
            # Get relevance scores from cross-encoder
            scores = self.model.predict(pairs)
            
            # Update documents with reranker scores
            reranked_docs = []
            for doc, score in zip(valid_docs, scores):
                # Create a copy to avoid modifying original
                reranked_doc = doc.copy()
                
                # Store both original and reranker scores
                reranked_doc['reranker_score'] = float(score)
                reranked_doc['original_score'] = doc.get('score', 0.0)
                
                # Update the main score with reranker score
                reranked_doc['score'] = float(score)
                
                # Add reranking method info
                if 'retrieval_method' in reranked_doc:
                    reranked_doc['retrieval_method'] += '_reranked'
                else:
                    reranked_doc['retrieval_method'] = 'reranked'
                
                reranked_docs.append(reranked_doc)
            
            # Sort by reranker score (descending)
            reranked_docs.sort(key=lambda x: x['reranker_score'], reverse=True)
            
            # Return top_k if specified
            if top_k:
                reranked_docs = reranked_docs[:top_k]
            
            print(f"Reranked {len(reranked_docs)} documents")
            
            # Log top scores for debugging
            if reranked_docs:
                top_scores = [doc['reranker_score'] for doc in reranked_docs[:3]]
                print(f"Top reranker scores: {top_scores}")
            
            return reranked_docs
            
        except Exception as e:
            print(f"Error during reranking: {e}")
            # Return original documents if reranking fails
            return documents
    
    def rerank_with_fusion(self, 
                          query: str, 
                          documents: List[Dict[str, Any]], 
                          alpha: float = 0.7,
                          top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using score fusion between original and reranker scores
        
        Args:
            query: The search query
            documents: List of documents to rerank
            alpha: Weight for reranker score (0-1), higher means more weight on reranker
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with fused scores
        """
        if not self.model or not documents:
            return documents
        
        try:
            # First get reranker scores
            reranked_docs = self.rerank_documents(query, documents, top_k=None)
            
            if not reranked_docs:
                return documents
            
            # Normalize original scores to 0-1 range
            original_scores = [doc.get('original_score', 0.0) for doc in reranked_docs]
            if max(original_scores) > 0:
                original_scores_norm = [s / max(original_scores) for s in original_scores]
            else:
                original_scores_norm = [0.0] * len(original_scores)
            
            # Normalize reranker scores to 0-1 range
            reranker_scores = [doc.get('reranker_score', 0.0) for doc in reranked_docs]
            min_reranker = min(reranker_scores)
            max_reranker = max(reranker_scores)
            
            if max_reranker > min_reranker:
                reranker_scores_norm = [(s - min_reranker) / (max_reranker - min_reranker) 
                                       for s in reranker_scores]
            else:
                reranker_scores_norm = [0.5] * len(reranker_scores)
            
            # Compute fused scores
            for i, doc in enumerate(reranked_docs):
                fused_score = (alpha * reranker_scores_norm[i] + 
                              (1 - alpha) * original_scores_norm[i])
                doc['fused_score'] = fused_score
                doc['score'] = fused_score  # Update main score
            
            # Sort by fused score
            reranked_docs.sort(key=lambda x: x['fused_score'], reverse=True)
            
            # Return top_k if specified
            if top_k:
                reranked_docs = reranked_docs[:top_k]
            
            print(f"Score fusion reranking completed for {len(reranked_docs)} documents")
            
            return reranked_docs
            
        except Exception as e:
            print(f"Error during fusion reranking: {e}")
            return documents
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'model_loaded': self.model is not None,
            'model_type': 'cross-encoder'
        } 