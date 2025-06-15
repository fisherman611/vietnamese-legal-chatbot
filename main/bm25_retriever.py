from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple
import pickle
import os
from utils.text_processor import VietnameseTextProcessor
from config import Config
from tqdm.auto import tqdm

os.makedirs("index", exist_ok=True)


class BM25Retriever:
    """BM25 retriever for initial document retrieval"""

    def __init__(self):
        self.text_processor = VietnameseTextProcessor()
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
        self.index_file = "index/bm25_index.pkl"

    def build_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents"""
        print("Building BM25 index...")

        self.documents = documents
        self.tokenized_corpus = []

        # Tokenize all documents
        for doc in tqdm(documents):
            content = doc.get("content", "")
            title = doc.get("title", "")

            # Combine title and content for better search
            full_text = f"{title} {content}"

            # Preprocess and tokenize
            processed_text = self.text_processor.preprocess_for_search(full_text)
            tokens = processed_text.split()

            self.tokenized_corpus.append(tokens)

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"BM25 index built with {len(self.documents)} documents")

    def save_index(self, filepath: str = None):
        """Save BM25 index to file"""
        if filepath is None:
            filepath = self.index_file

        try:
            index_data = {
                "bm25": self.bm25,
                "documents": self.documents,
                "tokenized_corpus": self.tokenized_corpus,
            }

            with open(filepath, "wb") as f:
                pickle.dump(index_data, f)

            print(f"BM25 index saved to {filepath}")
        except Exception as e:
            print(f"Error saving BM25 index: {e}")

    def load_index(self, filepath: str = None):
        """Load BM25 index from file"""
        if filepath is None:
            filepath = self.index_file

        try:
            if not os.path.exists(filepath):
                print(f"Index file {filepath} not found")
                return False

            with open(filepath, "rb") as f:
                index_data = pickle.load(f)

            self.bm25 = index_data["bm25"]
            self.documents = index_data["documents"]
            self.tokenized_corpus = index_data["tokenized_corpus"]

            print(f"BM25 index loaded from {filepath}")
            return True
        except UnicodeDecodeError as e:
            print(f"Encoding error loading BM25 index: {e}")
            print(f"Removing corrupted index file: {filepath}")
            try:
                os.remove(filepath)
            except:
                pass
            return False
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Corrupted BM25 index file: {e}")
            print(f"Removing corrupted index file: {filepath}")
            try:
                os.remove(filepath)
            except:
                pass
            return False
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            return False

    def search(
        self, query: str, top_k: int = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search documents using BM25"""
        if top_k is None:
            top_k = Config.BM25_TOP_K

        if self.bm25 is None:
            print("BM25 index not built. Please build index first.")
            return []

        # Preprocess query
        processed_query = self.text_processor.preprocess_for_search(query)
        query_tokens = processed_query.split()

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top documents with scores
        doc_score_pairs = [
            (self.documents[i], scores[i]) for i in range(len(self.documents))
        ]

        # Sort by score (descending)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = doc_score_pairs[:top_k]

        print(f"BM25 search returned {len(results)} results for query: {query}")
        return results

    def get_relevant_documents(
        self, query: str, top_k: int = None, min_score: float = None
    ) -> List[Dict[str, Any]]:
        """Get relevant documents above minimum score threshold"""
        # Set default minimum score for BM25 (more lenient than vector search)
        if min_score is None:
            min_score = 1.0  # Minimum BM25 score threshold
            
        results = self.search(query, top_k)

        # Filter by minimum score and enhance with score normalization
        filtered_results = []
        max_score = max([score for _, score in results]) if results else 0
        
        for doc, score in results:
            if score >= min_score:
                # Add normalized score for better comparison
                doc_with_score = doc.copy()
                doc_with_score['score'] = score
                doc_with_score['normalized_score'] = score / max_score if max_score > 0 else 0
                doc_with_score['retrieval_method'] = 'bm25'
                filtered_results.append(doc_with_score)

        print(f"BM25 found {len(filtered_results)} relevant documents (min_score={min_score})")
        return filtered_results

    def search_with_keywords(
        self, keywords: List[str], top_k: int = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search using multiple keywords"""
        # Combine keywords into a single query
        query = " ".join(keywords)
        return self.search(query, top_k)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 index"""
        if self.bm25 is None:
            return {}

        return {
            "total_documents": len(self.documents),
            "total_tokens": sum(len(tokens) for tokens in self.tokenized_corpus),
            "average_document_length": sum(
                len(tokens) for tokens in self.tokenized_corpus
            )
            / len(self.tokenized_corpus)
            if self.tokenized_corpus
            else 0,
            "vocabulary_size": len(
                set(token for tokens in self.tokenized_corpus for token in tokens)
            ),
        }
