"""简化的 WikiEval 评估测试"""
import asyncio
import pandas as pd
from pathlib import Path

from openai import OpenAI
from rag import ExampleRAG, SimpleKeywordRetriever

# 配置 SiliconFlow DeepSeek
api_key = "sk-cxwvirgzjrvlvleqxzwobedlbcetqrgtqyhsydylujozahnf"
openai_client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

print("=" * 60)
print("简化版 WikiEval 测试")
print("=" * 60)

# 1. 加载数据
print("\n1. 加载数据集...")
data_path = Path("e:/code/rag_test/WikiEval/data")
parquet_files = list(data_path.glob("*.parquet"))

if not parquet_files:
    print("[ERROR] 未找到数据文件")
    exit(1)

df = pd.read_parquet(parquet_files[0])
print(f"[OK] 加载了 {len(df)} 个样本")

# 2. 准备 RAG 知识库
print("\n2. 准备 RAG 知识库...")
documents = []
for i in range(min(5, len(df))):
    context_v1 = df.iloc[i]['context_v1']
    if context_v1:
        documents.extend(context_v1)

print(f"[OK] 知识库包含 {len(documents)} 个文档")

# 创建 RAG 系统
rag_client = ExampleRAG(
    llm_client=openai_client,
    retriever=SimpleKeywordRetriever(),
    logdir="evals/wikieval_logs",
    model_name="deepseek-ai/DeepSeek-V3.1-Terminus"
)
rag_client.add_documents(documents)

# 3. 运行测试
print("\n3. 运行测试查询...")
test_idx = 5  # 测试第 6 个样本
question = df.iloc[test_idx]['question']
ground_truth = df.iloc[test_idx]['answer']

print(f"\n问题: {question[:100]}...")
print(f"标准答案: {ground_truth[:100]}...")

# 执行查询
print("\n执行 RAG 查询...")
result = rag_client.query(question, top_k=3)

print(f"\nRAG 答案: {result['answer'][:200]}...")
print(f"\n日志文件: {result['logs']}")

# 4. 简单对比
print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
print(f"\n问题: {question}")
print(f"\n标准答案:\n{ground_truth}")
print(f"\nRAG 答案:\n{result['answer']}")
