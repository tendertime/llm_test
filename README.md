# RAG Test 项目

RAG系统测试和评估工具集。

## 项目结构

```
rag_test/
├── rag_eval/           # RAG评估工具
├── longbench/          # LongBench数据集处理
├── WikiEval/           # WikiEval相关
└── scripts/           # 辅助脚本
```

## 安装

```bash
pip install -r requirements.txt
```

## 配置

1. 复制环境变量配置文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的API密钥：
```
SILICONFLOW_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## 使用

### RAG评估
```bash
cd rag_eval
python eval.py
```

### LongBench数据集处理
```bash
python convert_longbench_to_ragas.py
```

## 许可证
MIT License
