import json
import pandas as pd
from typing import List, Dict, Any
from config import Config
from tqdm.auto import tqdm


class LegalDataLoader:
    """Load and process legal corpus"""

    def __init__(self):
        self.legal_corpus = None

    def load_legal_corpus(self) -> List[Dict[str, Any]]:
        """Load legal corpus from JSON file"""
        try:
            with open(Config.CORPUS_PATH, "r", encoding="utf-8") as f:
                self.legal_corpus = json.load(f)

            # Handle the case where the corpus is a list of laws with nested articles
            if isinstance(self.legal_corpus, list):
                print(f"Loaded {len(self.legal_corpus)} legal documents")
            else:
                # Handle single law document format
                print(
                    f"Loaded legal document: {self.legal_corpus.get('law_id', 'Unknown')}"
                )
                self.legal_corpus = [self.legal_corpus]

            return self.legal_corpus

        except FileNotFoundError:
            print(f"Legal corpus file not found at {Config.CORPUS_PATH}")
            return []
        except json.JSONEncoder as e:
            print(f"Error parsing JSON file: {e}")
            return []

    def prepare_documents_for_indexing(self) -> List[Dict[str, Any]]:
        """Prepare legal documents for vector indexing"""
        if self.legal_corpus is None:
            self.load_legal_corpus()

        documents = []
        for law in tqdm(self.legal_corpus):
            law_id = law.get("law_id", "")
            articles = law.get("articles", [])

            # Process each article in the law
            for article in articles:
                article_id = article.get("article_id", "")
                title = article.get("title", "")
                content = article.get("text", "")

                if content and content.strip():
                    # Create unique document ID combining law_id and article_id
                    doc_id = (
                        f"{law_id}_{article_id}"
                        if law_id and article_id
                        else article_id
                    )
                    documents.append(
                        {
                            "id": doc_id,
                            "title": title,
                            "content": content,
                            "metadata": {
                                "law_id": law_id,
                                "article_id": article_id,
                                "title": title,
                                "source": "legal_corpus",
                            },
                        }
                    )

        print(f"Prepared {len(documents)} documents for indexing")
        return documents

    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """Get a specific document by ID"""
        if self.legal_corpus is None:
            self.load_legal_corpus()

        # Handle both formats: "law_id_article_id" or just "article_id"
        for law in self.legal_corpus:
            law_id = law.get("law_id", "")
            articles = law.get("articles", [])

            for article in articles:
                article_id = article.get("article_id", "")
                combined_id = (
                    f"{law_id}_{article_id}" if law_id and article_id else article_id
                )

                if combined_id == doc_id or article_id == doc_id:
                    return {
                        "law_id": law_id,
                        "article_id": article_id,
                        "title": article.get("title", ""),
                        "text": article.get("text", ""),
                        "combined_id": combined_id,
                    }
        return {}

    def search_documents_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Search documents containing specific keywords"""
        if self.legal_corpus is None:
            self.load_legal_corpus()

        results = []
        keyword_lower = keyword.lower()

        for law in self.legal_corpus:
            law_id = law.get("law_id", "")
            articles = law.get("articles", [])

            for article in articles:
                content = article.get("text", "").lower()
                title = article.get("title", "").lower()

                if keyword_lower in content or keyword_lower in title:
                    article_id = article.get("article_id", "")
                    combined_id = (
                        f"{law_id}_{article_id}"
                        if law_id and article_id
                        else article_id
                    )

                    results.append(
                        {
                            "law_id": law_id,
                            "article_id": article_id,
                            "title": article.get("title", ""),
                            "text": article.get("text", ""),
                            "combined_id": combined_id,
                        }
                    )

        return results
