# LongBench RAGAS数据集使用指南

## 概述

本目录包含LongBench数据集转换为RAGAS评估格式的CSV文件，可用于评估RAG系统的性能。

## 文件说明

### 生成的数据集

- `multifieldqa_zh_top100.csv` - 中文多文档问答（前100条）
- `multifieldqa_zh.csv` - 中文多文档问答（100条）
- `multifieldqa_en.csv` - 英文多文档问答（50条）
- `passage_count.csv` - 段落计数（100条）
- `narrativeqa.csv` - 叙事问答（50条）
- `hotpotqa.csv` - HotpotQA（50条）
- `gov_report.csv` - 政府报告（50条）
- `qmsum.csv` - QMSum（50条）
- `combined_all.csv` - 合并所有数据集（150条）

### 数据格式

每个CSV文件包含以下列：

- **question**: 问题
- **contexts**: 上下文列表（字符串形式的Python列表）
- **ground_truth**: 真实答案列表
- **answer**: 生成的答案（需由RAG系统填充，初始为空）
- **dataset**: 原始数据集名称

## 使用方法

### 1. 加载数据

```python
import pandas as pd

# 加载数据集
df = pd.read_csv('e:/code/rag_test/longbench/ragas/multifieldqa_zh_top100.csv', encoding='utf-8-sig')

# 查看数据
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(df.head())
```

### 2. 准备RAGAS格式

```python
def prepare_for_ragas(df):
    """准备RAGAS评估所需的数据格式"""
    # 确保contexts是列表格式
    df['contexts'] = df['contexts'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # 确保ground_truth是列表格式
    df['ground_truth'] = df['ground_truth'].apply(lambda x: eval(x) if isinstance(x, str) else [x])
    
    return df

df = prepare_for_ragas(df)
```

### 3. 用RAG系统生成答案

```python
def generate_answers_with_rag(df, rag_system):
    """使用RAG系统生成答案"""
    answers = []
    
    for idx, row in df.iterrows():
        question = row['question']
        contexts = row['contexts']
        
        # 使用RAG系统生成答案
        answer = rag_system.generate(question, contexts)
        answers.append(answer)
    
    df['answer'] = answers
    return df

# 使用你的RAG系统
# df = generate_answers_with_rag(df, your_rag_system)

# 保存带答案的数据集
# df.to_csv('multifieldqa_zh_top100_with_answers.csv', index=False, encoding='utf-8-sig')
```

### 4. 运行RAGAS评估

```python
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# 转换为HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# 定义评估指标
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
]

# 运行评估
result = evaluate(dataset=dataset, metrics=metrics)

# 查看结果
print(result)

# 保存结果
result.to_pandas().to_csv('evaluation_results.csv', index=False)
```

## 评估指标说明

RAGAS提供以下评估指标：

1. **Faithfulness (忠实度)**: 生成答案相对于检索到的上下文的忠实程度
2. **Answer Relevancy (答案相关性)**: 生成答案与问题的相关程度
3. **Context Precision (上下文精确度)**: 检索到的上下文中包含相关信息比例
4. **Context Recall (上下文召回率)**: 检索到的上下文覆盖真实答案中信息的程度

## 数据集特点

### multifieldqa_zh (中文多文档问答)
- 长文本问答任务
- 需要从长文档中提取信息回答问题
- 适合测试RAG系统的长文档理解能力

### narrativeqa (叙事问答)
- 基于故事文本的问答
- 需要理解情节和细节
- 适合测试RAG系统的故事理解能力

### gov_report (政府报告)
- 官方文档问答
- 语言正式，信息密集
- 适合测试RAG系统的正式文档理解能力

### hotpotqa (多跳问答)
- 需要多次推理的问答
- 适合测试RAG系统的推理能力

## 注意事项

1. **answer列**: 初始为空，需要先用RAG系统生成答案才能进行评估
2. **contexts格式**: 存储为字符串形式的Python列表，使用时需要用`eval()`转换
3. **编码**: 文件使用UTF-8编码，加载时需要指定`encoding='utf-8-sig'`
4. **评估资源**: RAGAS评估需要LLM API（如OpenAI），确保有相应的API密钥

## 示例代码

完整的示例代码请参考：
- `convert_longbench_to_ragas.py` - 数据转换脚本
- `ragas_evaluation_example.py` - RAGAS评估示例

## 相关资源

- LongBench: https://github.com/THUDM/LongBench
- RAGAS: https://github.com/explodinggradients/ragas
- RAGAS文档: https://docs.ragas.io/

## 许可证

LongBench数据集遵循其原始许可证。
