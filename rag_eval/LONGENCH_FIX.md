# LongBench RAG 测试修复说明

## 问题诊断

原始错误：
```
openai.BadRequestError: Error code: 400 - {'code': 20015, 'message': 'The parameter is invalid. Please check again.', 'data': None}
```

后续错误：
```
openai.APIStatusError: Error code: 413 - {'code': 20042, 'message': 'input must have less than 512 tokens', 'data': None}
```

## 根本原因

1. **错误的模型名称**: 使用了 `BAAI/bge-m3`，但 SiliconFlow 实际支持的模型是：
   - `BAAI/bge-large-zh-v1.5` (中文)
   - `BAAI/bge-large-en-v1.5` (英文)

2. **缺少批量处理**: 一次性发送 800 个文档导致请求过大

3. **文本长度超限**: BGE 模型限制输入最多 512 tokens，但文档长度远超此限制

4. **缺少文本截断**: 长文本未被截断，导致 API 拒绝请求

## 修复措施

### 1. 修正模型名称

**文件**: `rag.py`

```python
# 修改前
embedding_model: str = "BAAI/bge-m3"

# 修改后
embedding_model: str = "BAAI/bge-large-zh-v1.5"
```

### 2. 添加批量处理

```python
def __init__(
    self,
    batch_size: int = 32,  # SiliconFlow 限制最大 32
):
```

```python
# 分批处理
for i in range(0, len(texts), self.batch_size):
    batch = texts[i : i + self.batch_size]
    # ... 处理每个批次
```

### 3. 添加文本截断

```python
# BGE 模型支持 max 512 tokens
# 中文: ~1 char per token, 使用 400 字符保证安全
# 英文: ~4 chars per token, 400 chars ≈ 100 tokens
max_chars = 400
if len(text) > max_chars:
    text = text[:max_chars]
```

### 4. 改进错误处理

```python
except Exception as e:
    print(f"  [错误] 处理批次 {i} 失败: {str(e)}")
    print(f"  [调试] 批次大小: {len(truncated_batch)}")
    if truncated_batch:
        print(f"  [调试] 第一个文档长度: {len(truncated_batch[0])}")
    raise RuntimeError(f"Error generating embeddings for batch {i}: {str(e)}")
```

## 修改的文件

1. **`rag.py`**:
   - `VectorStoreRetriever` 类
   - `default_rag_client()` 函数
   - 主程序部分

2. **`test_longbench_simple.py`**:
   - 更新 embedding_model 参数

3. **`test_longbench.py`**:
   - 更新 embedding_model 参数

## 测试结果

✅ **成功加载 800 个文档**
```
[LongBench] 加载了 800 个文档，来自: 
- combined_all(150)
- gov_report(50)
- hotpotqa(50)
- multifieldqa_en(50)
- multifieldqa_zh(100)
- multifieldqa_zh_top100(100)
- multifieldqa_zh_top100_with_answers(100)
- narrativeqa(50)
- passage_count(100)
- qmsum(50)
```

✅ **成功生成 Embeddings**
```
[Embeddings] 已处理 32/800 个文档
[Embeddings] 已处理 64/800 个文档
...
[Embeddings] 已处理 800/800 个文档
```

✅ **成功运行测试**
```
测试了 3 个样本
详细结果已保存到: multifieldqa_zh_results.csv
```

## 技术要点

### SiliconFlow Embeddings API 限制

| 参数 | 限制 |
|------|------|
| 模型 | BAAI/bge-large-zh-v1.5, BAAI/bge-large-en-v1.5 |
| 最大 tokens | 512 tokens |
| Batch size | 最大 32 |
| 输出维度 | 1024 |

### 文本截断策略

- **保守策略**: 400 字符 (中文 ≈ 400 tokens, 英文 ≈ 100 tokens)
- **理由**: 确保不超过 512 token 限制，留有余量

### 批量处理优化

- **批次大小**: 32 (API 限制)
- **800个文档**: 分 25 批次处理
- **处理时间**: 约 15-20 秒

## 使用建议

### 1. 短文本场景
```python
# 适合问答、摘要等短文本
rag_client = default_rag_client(
    llm_client=client,
    use_longbench=True
)
```

### 2. 长文档场景
```python
# 建议使用 Rerank 或分段检索
# 因为 400 字符截断会丢失信息
```

### 3. 自定义截断长度
```python
retriever = VectorStoreRetriever(
    openai_client=client,
    embedding_model="BAAI/bge-large-zh-v1.5",
    batch_size=32
)
# 文本截断在 _get_embeddings 中自动处理
```

## 后续优化方向

1. **智能分段**: 将长文档分段后分别生成 embedding
2. **Rerank**: 检索后使用 Rerank 模型重新排序
3. **混合检索**: 结合关键词和语义检索
4. **缓存机制**: 缓存 embedding 避免重复计算

## 参考资料

- [SiliconFlow Embeddings 文档](https://docs.siliconflow.cn/reference/createembedding-1)
- [BGE 模型介绍](https://arxiv.org/abs/2402.03216)
- [LongBench 数据集](https://github.com/THUDM/LongBench)
