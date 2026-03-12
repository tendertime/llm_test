import ast
import json
import numpy as np
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# 示例文档（用于测试）
DOCUMENTS = [
    "Ragas are melodic frameworks in Indian classical music.",
    "There are many types of ragas, each with its own mood and time of day.",
    "Ragas are used to evoke specific emotions in the listener.",
    "The performance of a raga involves improvisation within a set structure.",
    "Ragas can be performed on various instruments or sung vocally.",
]


@dataclass
class DocumentChunk:
    """文档分片数据结构"""
    chunk_id: int  # 分片ID
    document_id: int  # 原始文档ID
    content: str  # 分片内容
    start_pos: int  # 在原文中的起始位置
    end_pos: int  # 在原文中的结束位置
    is_first: bool  # 是否是第一个分片
    is_last: bool  # 是否是最后一个分片


class DocumentChunker:
    """
    文档分片器
    使用滑动窗口策略将长文档分割成适合embedding的小片段
    """

    def __init__(
        self,
        chunk_size: int = 350,
        chunk_overlap: int = 50,
        max_chunks_per_doc: int = 20,
    ):
        """
        初始化分片器

        Args:
            chunk_size: 每个分片的最大字符数（默认350，约等于350 tokens）
            chunk_overlap: 分片之间的重叠字符数（默认50）
            max_chunks_per_doc: 每个文档的最大分片数（防止文档过长）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks_per_doc = max_chunks_per_doc

    def split_document(self, document: str, document_id: int) -> List[DocumentChunk]:
        """
        将文档分割成多个分片

        Args:
            document: 文档内容
            document_id: 文档ID

        Returns:
            分片列表
        """
        if len(document) <= self.chunk_size:
            # 文档足够短，不需要分片
            return [
                DocumentChunk(
                    chunk_id=0,
                    document_id=document_id,
                    content=document,
                    start_pos=0,
                    end_pos=len(document),
                    is_first=True,
                    is_last=True,
                )
            ]

        chunks = []
        chunk_id = 0
        start = 0

        while start < len(document) and chunk_id < self.max_chunks_per_doc:
            # 计算当前分片的结束位置
            end = start + self.chunk_size

            # 如果不是最后一个分片，尝试在句号、问号、感叹号处分割
            if end < len(document):
                # 查找最近的句子结束符
                for i in range(end, max(start, end - 50), -1):
                    if document[i] in '。！？.!?\n':
                        end = i + 1
                        break

            # 提取分片内容
            chunk_content = document[start:end].strip()
            if not chunk_content:
                break

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=chunk_content,
                    start_pos=start,
                    end_pos=end,
                    is_first=(chunk_id == 0),
                    is_last=(end >= len(document)),
                )
            )

            # 移动到下一个分片（考虑重叠）
            start = end - self.chunk_overlap
            if start <= chunks[-1].start_pos:
                start = end  # 避免重复
            chunk_id += 1

        return chunks

    def split_documents(
        self, documents: List[str]
    ) -> Tuple[List[DocumentChunk], Dict[int, str]]:
        """
        批量分割文档

        Args:
            documents: 文档列表

        Returns:
            (分片列表, 原始文档映射)
        """
        all_chunks = []
        original_docs = {}

        for doc_id, doc in enumerate(documents):
            # 保存原始文档
            original_docs[doc_id] = doc

            # 分割文档
            chunks = self.split_document(doc, doc_id)
            all_chunks.extend(chunks)

        print(
            f"[文档分片] {len(documents)} 个文档 → {len(all_chunks)} 个分片 "
            f"(平均每个文档 {len(all_chunks)/len(documents):.1f} 个分片)"
        )

        return all_chunks, original_docs


def load_longbench_documents(
    longbench_dir: Optional[str] = None, dataset_names: Optional[List[str]] = None
) -> List[str]:
    """
    从 LongBench 数据集中加载所有 contexts 作为知识库文档

    Args:
        longbench_dir: LongBench 数据目录路径（默认为 ../longbench/ragas）
        dataset_names: 要加载的数据集名称列表（如 ['multifieldqa_zh', 'qmsum']）
                      如果为 None，则加载所有数据集

    Returns:
        文档列表
    """
    if longbench_dir is None:
        # 默认路径：相对于当前文件的上级目录
        longbench_dir = Path(__file__).parent.parent / "longbench" / "ragas"
    else:
        longbench_dir = Path(longbench_dir)

    if not longbench_dir.exists():
        print(f"警告: LongBench 目录不存在: {longbench_dir}")
        return []

    # 获取所有 CSV 文件
    if dataset_names:
        csv_files = [longbench_dir / f"{name}.csv" for name in dataset_names]
        csv_files = [f for f in csv_files if f.exists()]
    else:
        csv_files = list(longbench_dir.glob("*.csv"))

    all_documents = []
    loaded_datasets = []

    for csv_file in csv_files:
        try:
            import pandas as pd

            # 读取 CSV
            df = pd.read_csv(csv_file, encoding="utf-8-sig")

            # 提取 contexts
            doc_count = 0
            for idx, row in df.iterrows():
                contexts_str = row.get("contexts", "[]")

                # 解析 contexts
                try:
                    contexts = (
                        ast.literal_eval(contexts_str)
                        if isinstance(contexts_str, str)
                        else contexts_str
                    )
                except:
                    contexts = []

                # 添加到总列表
                for ctx in contexts:
                    if isinstance(ctx, str) and ctx.strip():
                        all_documents.append(ctx)
                        doc_count += 1

            loaded_datasets.append(f"{csv_file.stem}({doc_count})")

        except Exception as e:
            print(f"警告: 加载 {csv_file.name} 失败: {e}")
            continue

    if loaded_datasets:
        print(
            f"[LongBench] 加载了 {len(all_documents)} 个文档，来自: {', '.join(loaded_datasets)}"
        )

    return all_documents


@dataclass
class TraceEvent:
    """Single event in the RAG application trace"""

    event_type: str
    component: str
    data: Dict[str, Any]


class BaseRetriever:
    """
    Base class for retrievers.
    Subclasses should implement the fit and get_top_k methods.
    """

    def __init__(self):
        self.documents = []

    def fit(self, documents: List[str]):
        """Store the documents"""
        self.documents = documents

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Retrieve top-k most relevant documents for the query."""
        raise NotImplementedError("Subclasses should implement this method.")


class SimpleKeywordRetriever(BaseRetriever):
    """Ultra-simple keyword matching retriever"""

    def __init__(self):
        super().__init__()

    def _count_keyword_matches(self, query: str, document: str) -> int:
        """Count how many query words appear in the document"""
        query_words = query.lower().split()
        document_words = document.lower().split()
        matches = 0
        for word in query_words:
            if word in document_words:
                matches += 1
        return matches

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Get top k documents by keyword match count"""
        scores = []

        for i, doc in enumerate(self.documents):
            match_count = self._count_keyword_matches(query, doc)
            scores.append((i, match_count))

        # Sort by match count (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:k]


class VectorStoreRetriever(BaseRetriever):
    """
    Vector-based retriever using embeddings for semantic similarity search.
    Uses OpenAI-compatible embeddings API for encoding documents and queries.
    Supports document chunking for long documents.
    """

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        embedding_dimension: int = 1024,
        batch_size: int = 32,
        enable_chunking: bool = True,
        chunk_size: int = 350,
        chunk_overlap: int = 50,
    ):
        """
        Initialize vector store retriever.

        Args:
            openai_client: OpenAI-compatible client instance for generating embeddings
            embedding_model: Embedding model to use (default: BAAI/bge-large-zh-v1.5 for SiliconFlow)
            embedding_dimension: Dimension of embedding vectors (bge-large: 1024)
            batch_size: Number of texts to process in one API call (max 32 for SiliconFlow)
            enable_chunking: Whether to enable document chunking (default: True)
            chunk_size: Maximum size of each chunk in characters (default: 350)
            chunk_overlap: Overlap between chunks in characters (default: 50)
        """
        super().__init__()
        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.enable_chunking = enable_chunking

        # Document chunking
        self.chunker = (
            DocumentChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            if enable_chunking
            else None
        )

        # Storage
        self.document_embeddings: Optional[np.ndarray] = None
        self.chunks: List[DocumentChunk] = []  # All chunks
        self.original_documents: Dict[int, str] = {}  # Original docs mapping

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using OpenAI-compatible API.
        Processes texts in batches to avoid API limits.

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of embeddings (shape: [len(texts), embedding_dimension])
        """
        if not self.openai_client:
            raise ValueError("OpenAI client is required for generating embeddings")

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            try:
                # Call embeddings API
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model, input=batch
                )

                # Extract embeddings for this batch
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                print(
                    f"  [Embeddings] 已处理 {min(i + self.batch_size, len(texts))}/{len(texts)} 个文本块"
                )

            except Exception as e:
                print(f"  [错误] 处理批次 {i} 失败: {str(e)}")
                print(f"  [调试] 批次大小: {len(batch)}")
                if batch:
                    print(f"  [调试] 第一个文本块长度: {len(batch[0])}")
                    print(f"  [调试] 第一个文本块前100字符: {batch[0][:100]}")
                raise RuntimeError(f"Error generating embeddings for batch {i}: {str(e)}")

        return np.array(all_embeddings)

    def fit(self, documents: List[str]):
        """
        Store documents and generate their embeddings.
        If chunking is enabled, splits long documents into chunks.

        Args:
            documents: List of document texts
        """
        self.documents = documents

        if not documents:
            self.document_embeddings = None
            return

        if self.enable_chunking and self.chunker:
            # Split documents into chunks
            print(f"[VectorStore] 启用文档分片模式...")
            self.chunks, self.original_documents = self.chunker.split_documents(documents)

            # Store original documents
            self.documents = [self.original_documents[i] for i in range(len(documents))]

            # Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in self.chunks]
            self.document_embeddings = self._get_embeddings(chunk_texts)

            print(f"[VectorStore] 分片统计:")
            print(f"  - 原始文档数: {len(self.original_documents)}")
            print(f"  - 分片总数: {len(self.chunks)}")
            print(f"  - 平均每文档: {len(self.chunks)/len(documents):.1f} 个分片")
        else:
            # No chunking, use original documents
            print(f"[VectorStore] 使用原始文档模式（不分片）")
            self.document_embeddings = self._get_embeddings(documents)
            self.chunks = []
            self.original_documents = {i: doc for i, doc in enumerate(documents)}

    def _cosine_similarity(
        self, query_embedding: np.ndarray, doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between query and documents.

        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding matrix

        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(
            doc_embeddings, axis=1, keepdims=True
        )

        # Calculate cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        return similarities

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """
        Retrieve top-k most similar documents for the query.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of tuples (document_index, similarity_score)
            If chunking enabled, returns original document indices with max chunk score
        """
        if not self.documents or self.document_embeddings is None:
            return []

        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]

        # Calculate similarities
        similarities = self._cosine_similarity(
            query_embedding, self.document_embeddings
        )

        if self.enable_chunking and self.chunks:
            # Aggregate scores by original document
            doc_scores = {}
            for chunk_idx, score in enumerate(similarities):
                doc_id = self.chunks[chunk_idx].document_id
                if doc_id not in doc_scores or score > doc_scores[doc_id]['max_score']:
                    doc_scores[doc_id] = {
                        'max_score': score,
                        'best_chunk_idx': chunk_idx
                    }

            # Sort by max score
            sorted_docs = sorted(
                doc_scores.items(),
                key=lambda x: x[1]['max_score'],
                reverse=True
            )[:k]

            # Return (doc_id, score) tuples
            return [(doc_id, float(data['max_score'])) for doc_id, data in sorted_docs]
        else:
            # No chunking, return top-k document indices
            top_k_indices = np.argsort(similarities)[::-1][:k]
            return [(int(idx), float(similarities[idx])) for idx in top_k_indices]

    def get_chunk_info(self, doc_id: int) -> List[DocumentChunk]:
        """
        Get all chunks for a specific document.

        Args:
            doc_id: Document ID

        Returns:
            List of chunks belonging to the document
        """
        return [chunk for chunk in self.chunks if chunk.document_id == doc_id]


class ExampleRAG:
    """
    Simple RAG system that:
    1. accepts a llm client
    2. uses simple keyword matching to retrieve relevant documents
    3. uses the llm client to generate a response based on the retrieved documents when a query is made
    """

    def __init__(
        self,
        llm_client,
        retriever: Optional[BaseRetriever] = None,
        system_prompt: Optional[str] = None,
        logdir: str = "logs",
        model_name: str = "deepseek-ai/DeepSeek-V3.1-Terminus",
    ):
        """
        Initialize RAG system

        Args:
            llm_client: LLM client with a generate() method
            retriever: Document retriever (defaults to SimpleKeywordRetriever)
            system_prompt: System prompt template for generation
            logdir: Directory for trace log files
            model_name: Model name to use for generation
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.retriever = retriever or SimpleKeywordRetriever()
        self.system_prompt = (
            system_prompt
            or """Answer the following question based on the provided documents:
                                Question: {query}
                                Documents:
                                {context}
                                Answer:
                            """
        )
        self.documents = []
        self.is_fitted = False
        self.traces = []
        self.logdir = logdir

        # Create log directory if it doesn't exist
        os.makedirs(self.logdir, exist_ok=True)

        # Initialize tracing
        self.traces.append(
            TraceEvent(
                event_type="init",
                component="rag_system",
                data={
                    "retriever_type": type(self.retriever).__name__,
                    "system_prompt_length": len(self.system_prompt),
                    "logdir": self.logdir,
                },
            )
        )

    def add_documents(self, documents: List[str]):
        """Add documents to the knowledge base"""
        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="rag_system",
                data={
                    "operation": "add_documents",
                    "num_new_documents": len(documents),
                    "total_documents_before": len(self.documents),
                    "document_lengths": [len(doc) for doc in documents],
                },
            )
        )

        self.documents.extend(documents)
        # Refit retriever with all documents
        self.retriever.fit(self.documents)
        self.is_fitted = True

        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="retriever",
                data={
                    "operation": "fit_completed",
                    "total_documents": len(self.documents),
                    "retriever_type": type(self.retriever).__name__,
                },
            )
        )

    def set_documents(self, documents: List[str]):
        """Set documents (replacing any existing ones)"""
        old_doc_count = len(self.documents)

        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="rag_system",
                data={
                    "operation": "set_documents",
                    "num_new_documents": len(documents),
                    "old_document_count": old_doc_count,
                    "document_lengths": [len(doc) for doc in documents],
                },
            )
        )

        self.documents = documents
        self.retriever.fit(self.documents)
        self.is_fitted = True

        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="retriever",
                data={
                    "operation": "fit_completed",
                    "total_documents": len(self.documents),
                    "retriever_type": type(self.retriever).__name__,
                },
            )
        )

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for the query

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of dictionaries containing document info
        """
        if not self.is_fitted:
            raise ValueError(
                "No documents have been added. Call add_documents() or set_documents() first."
            )

        self.traces.append(
            TraceEvent(
                event_type="retrieval",
                component="retriever",
                data={
                    "operation": "retrieve_start",
                    "query": query,
                    "query_length": len(query),
                    "top_k": top_k,
                    "total_documents": len(self.documents),
                },
            )
        )

        top_docs = self.retriever.get_top_k(query, k=top_k)

        retrieved_docs = []
        for idx, score in top_docs:
            if score > 0:  # Only include documents with positive similarity scores
                retrieved_docs.append(
                    {
                        "content": self.documents[idx],
                        "similarity_score": score,
                        "document_id": idx,
                    }
                )

        self.traces.append(
            TraceEvent(
                event_type="retrieval",
                component="retriever",
                data={
                    "operation": "retrieve_complete",
                    "num_retrieved": len(retrieved_docs),
                    "scores": [doc["similarity_score"] for doc in retrieved_docs],
                    "document_ids": [doc["document_id"] for doc in retrieved_docs],
                },
            )
        )

        return retrieved_docs

    def generate_response(self, query: str, top_k: int = 3) -> str:
        """
        Generate response to query using retrieved documents

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Generated response
        """
        if not self.is_fitted:
            raise ValueError(
                "No documents have been added. Call add_documents() or set_documents() first."
            )

        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k)

        if not retrieved_docs:
            return "I couldn't find any relevant documents to answer your question."

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # Generate response using LLM client
        prompt = self.system_prompt.format(query=query, context=context)

        self.traces.append(
            TraceEvent(
                event_type="llm_call",
                component="siliconflow_api",
                data={
                    "operation": "generate_response",
                    "model": self.model_name,
                    "query": query,
                    "prompt_length": len(prompt),
                    "context_length": len(context),
                    "num_context_docs": len(retrieved_docs),
                },
            )
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            response_text = response.choices[0].message.content.strip()

            self.traces.append(
                TraceEvent(
                    event_type="llm_response",
                    component="siliconflow_api",
                    data={
                        "operation": "generate_response",
                        "response_length": len(response_text),
                        "usage": (
                            response.usage.model_dump() if response.usage else None
                        ),
                        "model": self.model_name,
                    },
                )
            )

            return response_text

        except Exception as e:
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="openai_api",
                    data={"operation": "generate_response", "error": str(e)},
                )
            )
            return f"Error generating response: {str(e)}"

    def query(
        self, question: str, top_k: int = 3, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents and generate response

        Args:
            question: User question
            top_k: Number of documents to retrieve
            run_id: Optional run ID for tracing (auto-generated if not provided)

        Returns:
            Dictionary containing response and retrieved documents
        """
        # Generate run_id if not provided
        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000:04d}"

        # Reset traces for this query
        self.traces = []

        self.traces.append(
            TraceEvent(
                event_type="query_start",
                component="rag_system",
                data={
                    "run_id": run_id,
                    "question": question,
                    "question_length": len(question),
                    "top_k": top_k,
                    "total_documents": len(self.documents),
                },
            )
        )

        try:
            retrieved_docs = self.retrieve_documents(question, top_k)
            response = self.generate_response(question, top_k)

            result = {"answer": response, "run_id": run_id}

            self.traces.append(
                TraceEvent(
                    event_type="query_complete",
                    component="rag_system",
                    data={
                        "run_id": run_id,
                        "success": True,
                        "response_length": len(response),
                        "num_retrieved": len(retrieved_docs),
                    },
                )
            )

            logs_path = self.export_traces_to_log(run_id, question, result)
            return {"answer": response, "run_id": run_id, "logs": logs_path}

        except Exception as e:
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="rag_system",
                    data={"run_id": run_id, "operation": "query", "error": str(e)},
                )
            )

            # Return error result
            logs_path = self.export_traces_to_log(run_id, question, None)
            return {
                "answer": f"Error processing query: {str(e)}",
                "run_id": run_id,
                "logs": logs_path,
            }

    def export_traces_to_log(
        self,
        run_id: str,
        query: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ):
        """Export traces to a log file with run_id"""
        timestamp = datetime.now().isoformat()
        log_filename = (
            f"rag_run_{run_id}_{timestamp.replace(':', '-').replace('.', '-')}.json"
        )
        log_filepath = os.path.join(self.logdir, log_filename)

        log_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "query": query,
            "result": result,
            "num_documents": len(self.documents),
            "traces": [asdict(trace) for trace in self.traces],
        }

        with open(log_filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"RAG traces exported to: {log_filepath}")
        return log_filepath


def default_rag_client(
    llm_client,
    logdir: str = "logs",
    model_name: str = "deepseek-ai/DeepSeek-V3.1-Terminus",
    embedding_model: str = "BAAI/bge-large-zh-v1.5",
    use_longbench: bool = True,
    longbench_datasets: Optional[List[str]] = None,
) -> ExampleRAG:
    """
    Create a default RAG client with SiliconFlow DeepSeek LLM and vector store retriever.

    Args:
        llm_client: LLM client instance (also used for embeddings)
        logdir: Directory for trace logs
        model_name: Model name to use (default: DeepSeek V3.1)
        embedding_model: Embedding model to use (default: BAAI/bge-large-zh-v1.5 for SiliconFlow)
        use_longbench: Whether to load LongBench documents (default: True)
        longbench_datasets: Specific LongBench datasets to load (default: all)

    Returns:
        ExampleRAG instance
    """
    retriever = VectorStoreRetriever(
        openai_client=llm_client, embedding_model=embedding_model
    )
    client = ExampleRAG(
        llm_client=llm_client, retriever=retriever, logdir=logdir, model_name=model_name
    )

    # 加载文档
    if use_longbench:
        # 加载 LongBench 知识库
        documents = load_longbench_documents(dataset_names=longbench_datasets)
        if documents:
            client.add_documents(documents)
        else:
            # 如果加载失败，使用默认文档
            print("[警告] LongBench 文档加载失败，使用默认示例文档")
            client.add_documents(DOCUMENTS)
    else:
        # 使用默认示例文档
        client.add_documents(DOCUMENTS)

    return client


if __name__ == "__main__":
    try:
        api_key = os.environ.get("SILICONFLOW_API_KEY", "sk-cxwvirgzjrvlvleqxzwobedlbcetqrgtqyhsydylujozahnf")
    except KeyError:
        print("Error: SILICONFLOW_API_KEY environment variable is not set.")
        print("Please set your SiliconFlow API key:")
        print("export SILICONFLOW_API_KEY='your_siliconflow_api_key'")
        exit(1)

    # Initialize RAG system with SiliconFlow DeepSeek
    llm = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1"
    )
    
    # Create vector store retriever using SiliconFlow BAAI/bge-large-zh-v1.5 embeddings
    retriever = VectorStoreRetriever(
        openai_client=llm,
        embedding_model="BAAI/bge-large-zh-v1.5",
        embedding_dimension=1024
    )
    rag_client = ExampleRAG(llm_client=llm, retriever=retriever, logdir="logs")

    # Add documents (this will be traced)
    #rag_client.add_documents(DOCUMENTS)

    # Run query with tracing
    query = "What is Ragas"
    print(f"Query: {query}")
    response = rag_client.query(query, top_k=3)

    print("Response:", response["answer"])
    print(f"Run ID: {response['logs']}")
