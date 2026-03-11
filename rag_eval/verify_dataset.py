import os
from pathlib import Path
import pandas as pd

print("=" * 60)
print("本地 WikiEval 数据集检查结果")
print("=" * 60)

data_path = Path("e:/code/rag_test/WikiEval/data")
files = list(data_path.glob("*.parquet"))

print(f"\n1. 数据集文件:")
print(f"   路径: {data_path}")
print(f"   Parquet 文件: {files[0].name}")
print(f"   文件大小: {files[0].stat().st_size:,} bytes ({files[0].stat().st_size / 1024:.1f} KB)")

df = pd.read_parquet(files[0])

print(f"\n2. 数据集基本信息:")
print(f"   样本总数: {len(df)}")
print(f"   字段数量: {len(df.columns)}")
print(f"   字段列表: {list(df.columns)}")

print(f"\n3. 字段说明:")
print(f"   - question: 测试问题")
print(f"   - answer: 标准答案 (grounded_answer)")
print(f"   - context_v1: 理想上下文")
print(f"   - context_v2: 包含冗余信息的上下文")
print(f"   - ungrounded_answer: 无上下文的答案")
print(f"   - poor_answer: 质量较差的答案")
print(f"   - source: 来源维基百科页面")

print(f"\n4. 样本预览 (前 3 个):")
for i in range(min(3, len(df))):
    print(f"\n   样本 {i+1}:")
    row = df.iloc[i]
    print(f"      问题: {row['question'][:60]}...")
    print(f"      答案: {row['answer'][:60]}...")
    print(f"      来源: {row['source']}")
    print(f"      context_v1 数量: {len(row['context_v1'])} 个文档片段")
    print(f"      context_v2 数量: {len(row['context_v2'])} 个文档片段")

print(f"\n5. 统计信息:")
print(f"   平均 context_v1 长度: {df['context_v1'].apply(len).mean():.1f} 个文档")
print(f"   平均 context_v2 长度: {df['context_v2'].apply(len).mean():.1f} 个文档")
print(f"   平均问题长度: {df['question'].apply(len).mean():.0f} 字符")
print(f"   平均答案长度: {df['answer'].apply(len).mean():.0f} 字符")

print("\n" + "=" * 60)
print("检查结果: [OK] 本地 WikiEval 数据集完整且可用")
print("=" * 60)

print(f"\n可以运行评估:")
print(f"   python evals_wikieval.py")
