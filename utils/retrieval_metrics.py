import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import time

from main.chatbot import VietnameseLegalRAG
from utils.data_loader import LegalDataLoader
from config import Config

class RAGEvaluator:
    def __init__(self):
        self.rag = VietnameseLegalRAG()
        self.setup_rag()
        
    def setup_rag(self):
        """Setup RAG system with documents"""
        print("🔧 Setting up RAG system...")
        loader = LegalDataLoader()
        documents = loader.prepare_documents_for_indexing()
        self.rag.setup_indices(documents, force_rebuild=False)
        print(f"✅ RAG system ready with {len(documents)} documents")
        
    def retrieve_documents_with_method(self, query: str, method: str ="hybrid_rerank", top_k: int =20) -> List[Dict[str, Any]]:
        """
        Retrieve documents using specific method
        
        Args:
            query: Search query
            method: One of 'bm25', 'vector', 'hybrid', 'hybrid_rerank'
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if method == "bm25":
            # Force BM25 only by termporarily disabling vector store
            original_vector_store = self.rag.vector_store
            self.rag.vector_store = None
            try:
                results = self.rag.retrieve_documents(query, use_hybrid=False, use_reranking=False)
            finally:
                self.rag.vector_store = original_vector_store
            return results[:top_k]
        
            return results[:top_k]
            
        elif method == "vector":
            # Force Vector DB only by temporarily disabling BM25
            original_bm25 = self.rag.bm25_retriever
            self.rag.bm25_retriever = None
            try:
                results = self.rag.retrieve_documents(query, use_hybrid=False, use_reranking=False)
            finally:
                self.rag.bm25_retriever = original_bm25
            return results[:top_k]
            
        elif method == "hybrid":
            return self.rag.retrieve_documents(query, use_hybrid=True, use_reranking=False)[:top_k]
            
        elif method == "hybrid_rerank":
            return self.rag.retrieve_documents(query, use_hybrid=True, use_reranking=True)[:top_k]
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bm25', 'vector', 'hybrid', or 'hybrid_rerank'")
        
    def calculate_precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
            
        retrieved_at_k = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc in retrieved_at_k if any(rel_doc in doc for rel_doc in relevant_docs))
        return relevant_retrieved / k
    
    def calculate_recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if len(relevant_docs) == 0:
            return 0.0
        
        retrieved_at_k = retrieved_docs[:k]
        relevant_retrieved = sum(1 for rel_doc in relevant_docs if any(rel_doc in ret_doc for ret_doc in retrieved_at_k))
        return relevant_retrieved / len(relevant_docs)
    
    def calculate_f1_at_k(self, precision_k: float, recall_k: float) -> float:
        """Calculate F1@K from Precision@K and Recall@K"""
        if precision_k + recall_k == 0:
            return 0.0
        return 2 * (precision_k * recall_k) / (precision_k + recall_k)
    
    def calculate_average_precision(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        if len(relevant_docs) == 0:
            return 0.0
        
        relevant_retrieved = 0
        precision_sum = 0.0
        
        for i, doc in enumerate(retrieved_docs):
            if any(rel_doc in doc for rel_doc in relevant_docs):
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs)
    
    def calculate_reciprocal_rank(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Reciprocal Rank for a single query"""
        for i, doc in enumerate(retrieved_docs):
            if any(rel_doc in doc for rel_doc in relevant_docs):
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_metrics_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k_values: List[int]) -> Dict[str, Dict[int, float]]:
        """Calculate all metrics at different K values"""
        metrics = {
            'precision': {},
            'recall': {},
            'f1': {}
        }
        
        for k in k_values:
            precision_k = self.calculate_precision_at_k(retrieved_docs, relevant_docs, k)
            recall_k = self.calculate_recall_at_k(retrieved_docs, relevant_docs, k)
            f1_k = self.calculate_f1_at_k(precision_k, recall_k)
            
            metrics['precision'][k] = precision_k
            metrics['recall'][k] = recall_k
            metrics['f1'][k] = f1_k
        
        return metrics
    
    def evaluate_retrieval_method(self, 
                                questions_df: pd.DataFrame,
                                method: str = "hybrid_rerank",
                                top_k: int = 100,
                                k_values: List[int] = None) -> Dict[str, Any]:
        """
        Evaluate retrieval performance for a specific method
        
        Args:
            questions_df: DataFrame with questions and relevant articles
            method: Retrieval method ('bm25', 'vector', 'hybrid', 'hybrid_rerank')
            top_k: Number of documents to retrieve
            k_values: K values for evaluation metrics
            
        Returns:
            Dictionary with detailed metrics
        """
        if k_values is None:
            k_values = [1, 10, 20]
        
        method_names = {
            'bm25': 'BM25 Only',
            'vector': 'Vector DB Only',
            'hybrid': 'Hybrid (BM25 + Vector)',
            'hybrid_rerank': 'Hybrid + Reranking'
        }
        
        print(f"🔍 Evaluating {method_names.get(method, method)}...")
        
        total_questions = len(questions_df)
        
        # Initialize metric containers
        hits_at_k = {k: 0 for k in k_values}
        precision_at_k = {k: [] for k in k_values}
        recall_at_k = {k: [] for k in k_values}
        f1_at_k = {k: [] for k in k_values}
        map_at_k = {k: [] for k in k_values}
        
        mrr_scores = []
        retrieval_times = []
        detailed_results = []
        
        for idx, row in tqdm(questions_df.iterrows(), total=total_questions, desc=f"Evaluating {method_names.get(method, method)}"):
            question = row['question']
            relevant_articles = eval(row['relevant_articles'])
            expected_doc_ids = set()
            
            # Build expected document IDs
            for article in relevant_articles:
                law_id = article['law_id']
                article_id = article['article_id']
                expected_doc_ids.add(f"{law_id}_{article_id}")
            
            # Measure retrieval time
            start_time = time.time()
            try:
                retrieved_docs = self.retrieve_documents_with_method(question, method, top_k)
            except Exception as e:
                print(f"Error retrieving documents for question {idx}: {e}")
                retrieved_docs = []
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)
            
            # Extract document IDs and scores
            retrieved_doc_ids = [doc.get('id', '') for doc in retrieved_docs[:top_k]]
            retrieved_scores = [doc.get('score', 0) for doc in retrieved_docs[:top_k]]
            
            # Find matches and positions
            found_positions = []
            matched_docs = []
            
            for i, doc_id in enumerate(retrieved_doc_ids):
                for expected_id in expected_doc_ids:
                    if expected_id in doc_id:
                        found_positions.append(i + 1)  # 1-indexed
                        matched_docs.append({
                            'position': i + 1,
                            'doc_id': doc_id,
                            'expected_id': expected_id,
                            'score': retrieved_scores[i]
                        })
                        break
            
            # Calculate metrics for this question at different K values
            query_metrics = self.calculate_metrics_at_k(retrieved_doc_ids, list(expected_doc_ids), k_values)
            
            # Add to aggregated metrics
            for k in k_values:
                precision_at_k[k].append(query_metrics['precision'][k])
                recall_at_k[k].append(query_metrics['recall'][k])
                f1_at_k[k].append(query_metrics['f1'][k])
                
                # MAP@K
                map_score_k = self.calculate_average_precision(retrieved_doc_ids[:k], list(expected_doc_ids))
                map_at_k[k].append(map_score_k)
                
                # Hits@K
                if found_positions and min(found_positions) <= k:
                    hits_at_k[k] += 1
            
            # MRR (overall, not @K)
            rr_score = self.calculate_reciprocal_rank(retrieved_doc_ids, list(expected_doc_ids))
            mrr_scores.append(rr_score)
            
            # Store detailed result
            detailed_results.append({
                'question_id': row.get('question_id', idx),
                'question': question,
                'expected_docs': list(expected_doc_ids),
                'retrieved_docs': retrieved_doc_ids[:10],  # Top 10
                'matched_docs': matched_docs,
                'found_positions': found_positions,
                'retrieval_time': retrieval_time,
                'metrics_at_k': query_metrics,
                'rr_score': rr_score,
                'method': method
            })
        
        # Calculate aggregate metrics
        metrics = {
            'method': method,
            'method_name': method_names.get(method, method),
            'total_questions': total_questions,
            'avg_retrieval_time': np.mean(retrieval_times),
            'median_retrieval_time': np.median(retrieval_times),
            'questions_with_results': sum(1 for r in detailed_results if r['found_positions']),
            'coverage': sum(1 for r in detailed_results if r['found_positions']) / total_questions,
            'mrr': np.mean(mrr_scores)
        }
        
        # Add K-specific metrics
        for k in k_values:
            metrics[f'hits_at_{k}'] = hits_at_k[k] / total_questions
            metrics[f'precision_at_{k}'] = np.mean(precision_at_k[k])
            metrics[f'recall_at_{k}'] = np.mean(recall_at_k[k])
            metrics[f'f1_at_{k}'] = np.mean(f1_at_k[k])
            metrics[f'map_at_{k}'] = np.mean(map_at_k[k])
            
            # Standard deviations for error analysis
            metrics[f'precision_at_{k}_std'] = np.std(precision_at_k[k])
            metrics[f'recall_at_{k}_std'] = np.std(recall_at_k[k])
            metrics[f'f1_at_{k}_std'] = np.std(f1_at_k[k])
            metrics[f'map_at_{k}_std'] = np.std(map_at_k[k])
        
        return {
            'metrics': metrics,
            'detailed_results': detailed_results,
            'k_values': k_values
        }
        
    def compare_all_retrieval_methods(self, questions_df: pd.DataFrame, k_values: List[int] = None) -> Dict[str, Any]:
        """Compare all four retrieval methods on the given dataset"""
        
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        print("⚖️ Comparing all retrieval methods...")
        
        methods = ['bm25', 'vector', 'hybrid', 'hybrid_rerank']
        method_names = {
            'bm25': 'BM25 Only',
            'vector': 'Vector DB Only', 
            'hybrid': 'Hybrid (BM25 + Vector)',
            'hybrid_rerank': 'Hybrid + Reranking'
        }
        
        comparison_results = {}
        detailed_results_all = {}
        
        for method in methods:
            print(f"\n📊 Testing: {method_names[method]}")
            
            try:
                result = self.evaluate_retrieval_method(
                    questions_df,
                    method=method,
                    k_values=k_values
                )
                
                comparison_results[method_names[method]] = result['metrics']
                detailed_results_all[method] = result['detailed_results']
                
                # Print summary
                metrics = result['metrics']
                print(f"   MRR: {metrics['mrr']:.4f}")
                for k in k_values:
                    if f'map_at_{k}' in metrics:
                        print(f"   MAP@{k}: {metrics[f'map_at_{k}']:.4f}")
                        print(f"   Precision@{k}: {metrics[f'precision_at_{k}']:.4f}")
                        print(f"   Recall@{k}: {metrics[f'recall_at_{k}']:.4f}")
                        print(f"   F1@{k}: {metrics[f'f1_at_{k}']:.4f}")
                        print(f"   Hits@{k}: {metrics[f'hits_at_{k}']:.4f}")
                        break  # Only show first k for summary
                print(f"   Coverage: {metrics['coverage']:.4f}")
                print(f"   Avg time: {metrics['avg_retrieval_time']:.3f}s")
                
            except Exception as e:
                print(f"   ❌ Error evaluating {method}: {e}")
                comparison_results[method_names[method]] = {'error': str(e)}
        
        return {
            'comparison_results': comparison_results,
            'detailed_results': detailed_results_all,
            'k_values': k_values
        }
    
    def print_comparison_table(self, comparison_results: Dict[str, Any], k_values: List[int]):
        """Print a formatted comparison table of all methods"""
        print(f"\n📊 RETRIEVAL METHODS COMPARISON")
        print("=" * 120)
        
        methods = list(comparison_results.keys())
        if not methods:
            print("No results to display")
            return
        
        # Header
        print(f"{'Method':<25} {'MRR':<8} {'Coverage':<10} {'Time (s)':<10}", end="")
        for k in k_values:
            print(f" P@{k:<4} R@{k:<4} F1@{k:<4} H@{k:<4}", end="")
        print()
        print("-" * 120)
        
        # Data rows
        for method, metrics in comparison_results.items():
            if 'error' in metrics:
                print(f"{method:<25} ERROR: {metrics['error']}")
                continue
                
            print(f"{method:<25} {metrics['mrr']:<8.4f} {metrics['coverage']:<10.4f} {metrics['avg_retrieval_time']:<10.3f}", end="")
            for k in k_values:
                p = metrics.get(f'precision_at_{k}', 0)
                r = metrics.get(f'recall_at_{k}', 0) 
                f1 = metrics.get(f'f1_at_{k}', 0)
                h = metrics.get(f'hits_at_{k}', 0)
                print(f" {p:<5.3f} {r:<5.3f} {f1:<5.3f} {h:<5.3f}", end="")
            print()
        
        print("=" * 120)
    
    def print_metrics_table(self, metrics: Dict[str, Any], k_values: List[int], title: str = "Retrieval Metrics"):
        """Print a formatted table of metrics"""
        print(f"\n📊 {title}")
        print("=" * 80)
        print(f"{'Metric':<15} {'K=1':<10} {'K=3':<10} {'K=5':<10} {'K=10':<10}")
        print("-" * 80)
        
        # Hits@K
        print(f"{'Hits@K':<15}", end="")
        for k in k_values:
            print(f"{metrics[f'hits_at_{k}']:<10.4f}", end="")
        print()
        
        # Precision@K
        print(f"{'Precision@K':<15}", end="")
        for k in k_values:
            print(f"{metrics[f'precision_at_{k}']:<10.4f}", end="")
        print()
        
        # Recall@K
        print(f"{'Recall@K':<15}", end="")
        for k in k_values:
            print(f"{metrics[f'recall_at_{k}']:<10.4f}", end="")
        print()
        
        # F1@K
        print(f"{'F1@K':<15}", end="")
        for k in k_values:
            print(f"{metrics[f'f1_at_{k}']:<10.4f}", end="")
        print()
        
        # MAP@K
        print(f"{'MAP@K':<15}", end="")
        for k in k_values:
            print(f"{metrics[f'map_at_{k}']:<10.4f}", end="")
        print()
        
        print("-" * 80)
        print(f"{'MRR:':<15} {metrics['mrr']:<10.4f}")
        print(f"{'Coverage:':<15} {metrics['coverage']:<10.4f}")
        print(f"{'Avg Time:':<15} {metrics['avg_retrieval_time']:<10.3f}s")
        print("=" * 80)
    
    def export_detailed_metrics(self, results: Dict[str, Any], filename: str = "detailed_metrics.csv"):
        """Export detailed per-question metrics to CSV for analysis"""
        detailed_results = results['detailed_results']
        k_values = results['k_values']
        
        rows = []
        for result in detailed_results:
            row = {
                'question_id': result['question_id'],
                'question': result['question'],
                'num_expected_docs': len(result['expected_docs']),
                'num_retrieved_docs': len(result['retrieved_docs']),
                'found_positions': str(result['found_positions']),
                'retrieval_time': result['retrieval_time'],
                'rr_score': result['rr_score']
            }
            
            # Add metrics for each K
            for k in k_values:
                row[f'precision_at_{k}'] = result['metrics_at_k']['precision'][k]
                row[f'recall_at_{k}'] = result['metrics_at_k']['recall'][k]
                row[f'f1_at_{k}'] = result['metrics_at_k']['f1'][k]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"📄 Detailed metrics exported to {filename}")
        

def main():
    """Evaluate all retrieval methods on train_qna.csv dataset"""
    
    print("🚀 Starting RAG evaluation on train_qna.csv dataset...\n")
    
    # Check if splits exist
    train_dir = "data/train"
    train_file = os.path.join(train_dir, "train_qna.csv")
    
    if not os.path.exists(train_file):
        print("❌ train_qna.csv not found. Please run split_dataset.py first.")
        return
    
    # Load train dataset
    print("📚 Loading train dataset...")
    train_df = pd.read_csv(train_file)
    print(f"   Training: {len(train_df)} examples")
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Evaluate all retrieval methods on train set
    print(f"\n{'='*80}")
    print("EVALUATING ALL RETRIEVAL METHODS ON TRAIN DATASET")
    print(f"{'='*80}")
    
    k_values = [1, 10, 20]
    
    # Run comprehensive comparison
    train_comparison = evaluator.compare_all_retrieval_methods(train_df, k_values=k_values)
    
    # Display results in a nice table format
    if 'comparison_results' in train_comparison:
        evaluator.print_comparison_table(train_comparison['comparison_results'], k_values)
    
    # Save comprehensive results
    output_file = f"evaluation_results_{Config.COLLECTION_NAME}.json"
    final_results = {
        'train_comparison_results': train_comparison['comparison_results'],
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'dataset': 'train_qna.csv',
            'total_questions': len(train_df)
        },
        'k_values_used': k_values,
        'methods_evaluated': ['BM25 Only', 'Vector DB Only', 'Hybrid (BM25 + Vector)', 'Hybrid + Reranking']
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Export detailed per-question metrics for each method
    if 'detailed_results' in train_comparison:
        for method, detailed_results in train_comparison['detailed_results'].items():
            if detailed_results:  # Check if results exist
                # Create a mock results structure for the export function
                mock_results = {
                    'detailed_results': detailed_results,
                    'k_values': k_values
                }
                filename = f"detailed_{method}_{Config.COLLECTION_NAME}.csv"
                evaluator.export_detailed_metrics(mock_results, filename)
    
    # Print summary recommendations
    print(f"\n💡 Summary & Recommendations:")
    
    if 'comparison_results' in train_comparison:
        comparison_results = train_comparison['comparison_results']
        
        # Find best method based on MRR
        valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
        if valid_results:
            best_method = max(valid_results.items(), key=lambda x: x[1]['mrr'])
            print(f"   🏆 Best method: {best_method[0]} (MRR: {best_method[1]['mrr']:.4f})")
            
            # Performance analysis for best method
            best_metrics = best_method[1]
            if best_metrics['hits_at_1'] < 0.3:
                print("   ⚠️  Low Hits@1 - consider lowering similarity thresholds")
            if best_metrics['coverage'] < 0.7:
                print("   ⚠️  Low coverage - many questions return no results")
            if best_metrics['avg_retrieval_time'] > 2.0:
                print("   ⚠️  Slow retrieval - consider optimizing indices")
            
            if best_metrics['mrr'] > 0.5:
                print("   ✅ Good retrieval performance!")
            
            # Compare methods
            print("\n📈 Method Analysis:")
            for method, metrics in valid_results.items():
                print(f"   {method}: MRR={metrics['mrr']:.4f}, Coverage={metrics['coverage']:.4f}, Time={metrics['avg_retrieval_time']:.3f}s")
        else:
            print("   ❌ No valid results found for analysis")

if __name__ == "__main__":
    main()