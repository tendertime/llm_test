"""快速测试改进版评估方法"""
import pandas as pd
from pathlib import Path
from openai import OpenAI
from rag import ExampleRAG, SimpleKeywordRetriever

print("=" * 70)
print("快速测试：改进版 WikiEval 评估")
print("=" * 70)

# 配置
api_key = "sk-cxwvirgzjrvlvleqxzwobedlbcetqrgtqyhsydylujozahnf"
openai_client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

# 加载数据
print("\n1. 加载数据集...")
data_path = Path("e:/code/rag_test/WikiEval/data")
parquet_files = list(data_path.glob("*.parquet"))
df = pd.read_parquet(parquet_files[0])
print(f"[OK] 加载了 {len(df)} 个样本")

# 测试前 3 个样本
print("\n2. 测试前 3 个样本...")
results = []

for i in range(min(3, len(df))):
    sample = df.iloc[i]
    question = sample['question']
    ground_truth = sample['answer']
    context_v1 = sample['context_v1']
    source = sample['source']

    print(f"\n样本 {i+1}: {source}")
    print(f"问题: {question[:80]}...")

    # 创建 RAG 系统
    rag_client = ExampleRAG(
        llm_client=openai_client,
        retriever=SimpleKeywordRetriever(),
        logdir="evals/quick_test_logs",
        model_name="deepseek-ai/DeepSeek-V3.1-Terminus"
    )

    # 添加该样本的上下文
    rag_client.add_documents(context_v1)

    # 执行查询
    result = rag_client.query(question, top_k=3)

    answer = result['answer']
    print(f"RAG 答案: {answer[:120]}...")

    # 简单评估
    if len(answer) > 50 and "cannot answer" not in answer.lower():
        score = "pass"
    else:
        score = "fail"

    print(f"评分: {score}")

    results.append({
        'index': i,
        'source': source,
        'question': question,
        'ground_truth': ground_truth,
        'rag_answer': answer,
        'score': score
    })

# 汇总结果
print("\n" + "=" * 70)
print("测试结果汇总")
print("=" * 70)

total = len(results)
passed = sum(1 for r in results if r['score'] == 'pass')
failed = total - passed

print(f"\n总样本数: {total}")
print(f"通过: {passed}")
print(f"失败: {failed}")
if total > 0:
    print(f"通过率: {passed/total*100:.1f}%")

# 详细结果
print("\n详细结果:")
for r in results:
    print(f"\n样本 {r['index']+1} ({r['source']}):")
    print(f"  问题: {r['question'][:80]}...")
    print(f"  RAG 答案: {r['rag_answer'][:150]}...")
    print(f"  评分: {r['score']}")

print("\n" + "=" * 70)
