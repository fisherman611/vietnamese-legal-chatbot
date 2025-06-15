import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
import numpy as np
from config import Config
from tqdm import tqdm


class QdrantVectorStore:
    """QDrant vector store for legal document embeddings"""

    def __init__(self):
        self.client = None
        self.embedding_model = None
        self.collection_name = Config.COLLECTION_NAME
        self._initialize_client()
        self._initialize_embedding_model()

    def _initialize_client(self):
        """Initialize QDrant client"""
        try:
            if Config.QDRANT_URL and Config.QDRANT_API_KEY:
                # Cloud QDrant
                self.client = QdrantClient(
                    url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY
                )
            else:
                # Local QDrant (fallback)
                self.client = QdrantClient(host="localhost", port=6333)

            print("QDrant client initialized successfully")
        except Exception as e:
            print(f"Error initializing QDrant client: {e}")
            raise

    def _initialize_embedding_model(self):
        """Initialize embedding model"""
        try:
            # Clear any potentially corrupted cache
            import tempfile
            import shutil

            cache_dir = os.path.join(tempfile.gettempdir(), "sentence_transformers")

            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            print(f"Embedding model {Config.EMBEDDING_MODEL} loaded successfully")
        except UnicodeDecodeError as e:
            print(f"Encoding error loading embedding model: {e}")
            print("Trying to clear sentence-transformers cache...")
            try:
                import tempfile
                import shutil

                cache_dir = os.path.join(tempfile.gettempdir(), "sentence_transformers")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    print("Cache cleared, retrying...")

                self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
                print(
                    f"Embedding model {Config.EMBEDDING_MODEL} loaded successfully after cache clear"
                )
            except Exception as retry_e:
                print(
                    f"Failed to load embedding model even after cache clear: {retry_e}"
                )
                raise
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

    def create_collection(self, vector_size: int = 384, force_recreate: bool = False):
        """Create collection in QDrant"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )
            print(f"Collection exists: {collection_exists}")

            if collection_exists:
                if force_recreate:
                    print(f"Force recreating collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                    print(f"Deleted existing collection: {self.collection_name}")
                    # Create collection
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=vector_size, distance=Distance.COSINE
                        ),
                    )
                    print(f"Successfully created collection: {self.collection_name}")
                else:
                    print(
                        f"Collection {self.collection_name} already exists - skipping creation"
                    )
                    return
            else:
                print(
                    f"Collection {self.collection_name} does not exist - creating new collection"
                )
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )
                print(f"Successfully created collection: {self.collection_name}")

        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to vector store"""
        try:
            points = []

            for doc in tqdm(documents):
                # Generate embedding
                content = doc.get("content", "")
                if not content:
                    continue

                embedding = self.embed_text(content)
                if not embedding:
                    continue

                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "article_id": doc.get("id", ""),
                        "title": doc.get("title", ""),
                        "content": content,
                        "metadata": doc.get("metadata", {}),
                    },
                )
                points.append(point)

            # Batch upload
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)
                print(
                    f"Uploaded batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}"
                )

            print(f"Successfully added {len(points)} documents to vector store")

        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

    def search_similar_documents(
        self, query: str, top_k: int = None, score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        if score_threshold is None:
            score_threshold = Config.SIMILARITY_THRESHOLD

        try:
            # Generate query embedding
            query_embedding = self.embed_text(query)
            if not query_embedding:
                return []

            # Search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
            )

            # Format results
            scores = []
            for result in search_results:
                scores.append(result.score)
            
            max_score = max(scores)
            min_score = min(scores)
            
            results = []
            for result in search_results:
                results.append(
                    {
                        "id": result.payload.get("article_id", ""),
                        "title": result.payload.get("title", ""),
                        "content": result.payload.get("content", ""),
                        "score": float((result.score - min_score) / (max_score - min_score)) if max_score > 0 and min_score > 0 and max_score != min_score else 0,
                        "metadata": result.payload.get("metadata", {}),
                    }
                )

            print(f"Vector DB found {len(results)} similar documents")
            return results

        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            info = self.client.get_collection(self.collection_name)
            result = {
                "name": self.collection_name,  # Use the collection name we know
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
            }
            print(f"Collection info: {result}")
            return result
        except Exception as e:
            print(f"Collection '{self.collection_name}' does not exist: {e}")
            return {}

    def delete_collection(self):
        """Delete collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
