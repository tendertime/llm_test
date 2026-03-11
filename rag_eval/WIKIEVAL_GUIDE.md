# WikiEval 数据集使用指南

本文档介绍如何使用 WikiEval 数据集来评估 RAG 系统。

## 📋 概述

**WikiEval** 是一个基于维基百科的高质量 RAG 评估数据集，包含：
- 50 个问题
- 每个问题的标准答案（grounded_answer）
- 相关的上下文文档（context_v1, context_v2）
- 不同质量的答案示例（grounded, ungrounded, poor）

## 🚀 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 下载 WikiEval 数据集

```bash
# 下载数据集到本地
uv run python prepare_wikieval.py
```

这将：
- 从 Hugging Face 下载 WikiEval 数据集
- 保存到 `e:/code/rag_test/WikiEval/data/`
- 显示数据集的基本信息和样本预览

### 3. 验证数据集

```bash
# 检查本地数据集
uv run python prepare_wikieval.py check
```

### 4. 运行评估

```bash
# 使用 WikiEval 数据集运行 RAG 评估
uv run python evals_wikieval.py
```

## 📊 工作流程

```
┌─────────────────────────────────────────────────┐
│ 1. 加载 WikiEval 数据集                         │
│    - 尝试从本地加载                             │
│    - 如果本地没有，从 Hugging Face 下载        │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 2. 准备 RAG 系统知识库                          │
│    - 使用前 5 个样本的 context_v1 作为文档      │
│    - 构建文档检索索引                           │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 3. 创建评估数据集                                │
│    - 使用后续 3 个样本作为测试问题              │
│    - 保存为 Ragas Dataset 格式                  │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 4. 运行评估实验                                 │
│    - 对每个测试问题：                           │
│      a. RAG 系统检索相关文档                    │
│      b. 生成答案                                │
│      c. 与标准答案对比评分                      │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ 5. 生成报告                                     │
│    - 显示通过/失败统计                          │
│    - 每个样本的详细结果                         │
│    - 保存到 CSV 文件                            │
└─────────────────────────────────────────────────┘
```

## 📁 输出文件

评估完成后，会生成以下文件：

### 1. 日志文件 (`evals/wikieval_logs/`)
包含每个 RAG 查询的详细追踪信息：
- 查询问题
- 检索到的文档
- 生成的答案
- 性能指标

### 2. 实验结果 (`evals/experiments/wikieval_eval.csv`)
CSV 格式的评估结果，包含：
- 问题
- 标准答案
- RAG 生成的答案
- 评分（pass/fail）
- 日志文件路径

## 🔧 自定义配置

### 修改知识库大小

在 `evals_wikieval.py` 中修改：

```python
# 使用更多样本文本构建知识库
rag_client = prepare_rag_with_wikieval(wikieval_dataset, sample_size=10)
```

### 修改测试样本数量

在 `evals_wikieval.py` 中修改：

```python
# 测试更多样本
eval_dataset = create_evaluation_dataset(wikieval_dataset, sample_size=10)
```

### 修改评估指标

```python
# 在 evals_wikieval.py 中定义新的评估指标
custom_metric = DiscreteMetric(
    name="my_custom_metric",
    prompt="你的评估提示词...",
    allowed_values=["pass", "fail"],
)
```

### 修改检索器

在 `evals_wikieval.py` 中：

```python
# 替换为其他检索器
from my_retrievers import VectorRetriever
retriever = VectorRetriever()
rag_client = ExampleRAG(llm_client=openai_client, retriever=retriever)
```

## 📈 评估结果示例

```
============================================================
WikiEval 数据集 RAG 系统评估
============================================================

正在从本地加载 WikiEval 数据集...
✓ 成功从本地加载数据集，共 50 个样本

正在准备 RAG 系统（使用前 5 个样本的 context_v1 作为文档库）...
✓ 文档库包含 23 个文档片段

正在创建评估数据集（使用 3 个测试样本）...
✓ 评估数据集包含 3 个样本

开始运行评估实验...

============================================================
实验完成！
============================================================

📊 评估结果摘要:
------------------------------------------------------------
总样本数: 3
通过: 2
失败: 1
通过率: 66.7%

📝 详细结果:
------------------------------------------------------------

样本 1:
  问题: What is the capital of France?
  标准答案: Paris is the capital and most populous city...
  RAG 答案: Paris is the capital of France...
  评分: pass
  日志文件: evals/wikieval_logs/rag_run_xxx.json
```

## 🐛 故障排除

### 问题 1: datasets 包未安装

```bash
uv add datasets
```

### 问题 2: 无法连接 Hugging Face

检查网络连接，或使用代理设置。

### 问题 3: 本地数据集加载失败

重新下载数据集：

```bash
rm -rf e:/code/rag_test/WikiEval/data
uv run python prepare_wikieval.py
```

### 问题 4: API 密钥未设置

```bash
# SiliconFlow DeepSeek (recommended)
export SILICONFLOW_API_KEY="your-api-key"

# Or OpenAI
export OPENAI_API_KEY="your-api-key"
```

## 📚 相关资源

- [WikiEval 数据集](https://huggingface.co/datasets/explodinggradients/wikieval)
- [Ragas 文档](https://docs.ragas.io)
- [项目 README](../README.md)
