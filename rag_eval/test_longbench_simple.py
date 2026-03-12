"""
简化的 LongBench 快速测试脚本
直接使用 RAG 系统测试 LongBench 数据，无需 RAGAS 框架
"""
import ast
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

from rag import default_rag_client

# 配置
DATASET_NAME = "multifieldqa_zh"  # 数据集名称
SAMPLE_SIZE = 3  # 测试样本数量

print("=" * 70)
print(f"LongBench 快速测试 - 数据集: {DATASET_NAME}")
print("=" * 70)

# 1. 加载数据
print(f"\n[1/3] 加载数据集...")
data_path = Path(__file__).parent.parent / "longbench" / "ragas" / f"{DATASET_NAME}.csv"

if not data_path.exists():
    print(f"[ERROR] 数据集文件不存在: {data_path}")
    exit(1)

df = pd.read_csv(data_path, encoding='utf-8-sig')
df = df.head(SAMPLE_SIZE)
print(f"[OK] 加载了 {len(df)} 个样本")

# 2. 初始化 RAG 系统（自动加载 LongBench 知识库）
print(f"\n[2/3] 初始化 RAG 系统（加载 LongBench 知识库）...")
api_key = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-cxwvirgzjrvlvleqxzwobedlbcetqrgtqyhsydylujozahnf"
)
openai_client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)

rag_client = default_rag_client(
    llm_client=openai_client,
    logdir="evals/longbench_quick_test",
    model_name="deepseek-ai/DeepSeek-V3.1-Terminus",
    embedding_model="BAAI/bge-large-zh-v1.5",  # SiliconFlow embeddings 模型
    use_longbench=True,  # 自动加载 LongBench 知识库
    longbench_datasets=None  # None = 加载所有数据集，或指定 ['multifieldqa_zh', 'qmsum']
)
print("[OK] RAG 系统已初始化")

# 3. 测试样本
print(f"\n[3/3] 测试 {len(df)} 个样本...")
print("-" * 70)

results = []

for idx, row in df.iterrows():
    print(f"\n样本 {idx + 1}/{len(df)}:")

    # 获取问题和真实答案
    question = row['question']
    ground_truth = row.get('ground_truth', '')

    # 处理 ground_truth 格式
    if isinstance(ground_truth, str):
        try:
            gt_list = ast.literal_eval(ground_truth)
            if isinstance(gt_list, list) and len(gt_list) > 0:
                ground_truth = gt_list[0]
        except:
            pass

    print(f"问题: {question[:150]}...")
    print(f"真实答案: {ground_truth[:150]}...")

    # 执行 RAG 查询（使用全局 LongBench 知识库）
    print("\n执行 RAG 查询...")
    result = rag_client.query(question, top_k=3)

    rag_answer = result.get('answer', '')
    print(f"RAG 答案: {rag_answer[:200]}...")

    # 记录结果
    results.append({
        'question': question,
        'ground_truth': ground_truth,
        'rag_answer': rag_answer,
        'log_file': result.get('logs', '')
    })

    print("-" * 70)

# 4. 显示摘要
print("\n" + "=" * 70)
print("测试完成!")
print("=" * 70)

print(f"\n测试了 {len(results)} 个样本")
print("\n结果摘要:")
for i, r in enumerate(results, 1):
    print(f"\n{i}. 问题: {r['question'][:80]}...")
    print(f"   真实答案: {r['ground_truth'][:80]}...")
    print(f"   RAG答案: {r['rag_answer'][:80]}...")

# 可选：保存结果到 CSV
output_file = Path("evals/longbench_quick_test") / f"{DATASET_NAME}_results.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n详细结果已保存到: {output_file.resolve()}")
