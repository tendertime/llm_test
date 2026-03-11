"""
改进版 WikiEval 评估
使用每个样本自己的 context_v1 作为知识库来评估该样本的问题
"""
import asyncio
import os
import json
from pathlib import Path
from datetime import datetime

from openai import OpenAI

# 导入 ragas 组件
from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

# 导入我们的 RAG 系统
from rag import ExampleRAG, SimpleKeywordRetriever

# 导入 Hugging Face 数据集库
try:
    from datasets import load_from_disk, load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# 配置 SiliconFlow DeepSeek 客户端
api_key = os.environ.get("SILICONFLOW_API_KEY", "sk-cxwvirgzjrvlvleqxzwobedlbcetqrgtqyhsydylujozahnf")
openai_client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)
llm = llm_factory("deepseek-ai/DeepSeek-V3.1-Terminus", client=openai_client)


def load_wikieval_dataset(wikieval_path: str = "e:/code/rag_test/WikiEval/data"):
    """从本地加载 WikiEval 数据集"""
    import pandas as pd

    wikieval_path = Path(wikieval_path)

    # 从本地 parquet 文件加载
    if wikieval_path.exists():
        print(f"正在从本地加载 WikiEval 数据集: {wikieval_path}")
        try:
            parquet_files = list(wikieval_path.glob("*.parquet"))
            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
                print(f"[OK] 成功从本地加载数据集，共 {len(df)} 个样本")
                return df
            else:
                print("[WARN] 未找到 parquet 文件")
        except Exception as e:
            print(f"[WARN] 本地加载失败: {e}")

    return None


def create_evaluation_dataset(wikieval_dataset, sample_size=5):
    """创建评估数据集"""
    print(f"\n正在创建评估数据集（使用 {sample_size} 个测试样本）...")

    dataset = Dataset(
        name="wikieval_eval_fixed",
        backend="local/csv",
        root_dir="evals",
    )

    # 选择样本进行测试
    start_idx = 0
    end_idx = min(start_idx + sample_size, len(wikieval_dataset))

    for i in range(start_idx, end_idx):
        sample = wikieval_dataset.iloc[i]
        row = {
            "question": sample['question'],
            "ground_truth": sample['answer'],
            "source": sample['source'],
            "context_v1": sample['context_v1'],  # 保存上下文
            "sample_index": i
        }
        dataset.append(row)

    dataset.save()
    print(f"[OK] 评估数据集已创建，包含 {end_idx - start_idx} 个样本")

    return dataset


# 定义评估指标
answer_quality_metric = DiscreteMetric(
    name="answer_quality",
    prompt="""评估生成的答案质量。

    问题: {question}
    标准答案: {ground_truth}
    生成的答案: {response}

    请比较生成答案和标准答案，评估：
    1. 是否准确回答了问题
    2. 内容是否相关和准确
    3. 是否包含关键信息

    返回 'pass' 如果答案质量好，返回 'fail' 如果答案质量差或有明显错误。
    """,
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_wikieval_experiment(row):
    """
    运行 WikiEval 实验
    为每个问题创建独立的 RAG 系统，使用该样本的 context_v1 作为知识库
    """
    # 提取样本数据
    context_v1 = row.get('context_v1', [])

    if not context_v1:
        print(f"[WARN] 样本 {row.get('sample_index')} 没有 context_v1，跳过")
        return {
            **row,
            "response": "No context available",
            "score": "fail",
            "log_file": "",
        }

    # 创建独立的 RAG 系统，使用该样本的上下文作为知识库
    rag_client = ExampleRAG(
        llm_client=openai_client,
        retriever=SimpleKeywordRetriever(),
        logdir="evals/wikieval_fixed_logs",
        model_name="deepseek-ai/DeepSeek-V3.1-Terminus"
    )

    # 添加文档到知识库
    rag_client.add_documents(context_v1)

    # 执行查询
    response = rag_client.query(row["question"], top_k=3)

    # 使用指标评分
    score = answer_quality_metric.score(
        llm=llm,
        question=row["question"],
        ground_truth=row["ground_truth"],
        response=response.get("answer", ""),
    )

    # 返回实验结果
    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "score": score.value,
        "log_file": response.get("logs", ""),
    }
    return experiment_view


async def main():
    """主函数：执行 WikiEval 评估"""
    print("=" * 60)
    print("WikiEval 数据集 RAG 系统评估 (改进版)")
    print("=" * 60)

    # 1. 加载 WikiEval 数据集
    wikieval_dataset = load_wikieval_dataset()
    if wikieval_dataset is None:
        return

    # 2. 创建评估数据集
    eval_dataset = create_evaluation_dataset(wikieval_dataset, sample_size=5)

    # 3. 运行实验
    print(f"\n开始运行评估实验...")
    experiment_results = await run_wikieval_experiment.arun(eval_dataset)

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)

    # 显示结果摘要
    print("\n评估结果摘要:")
    print("-" * 60)

    total = len(experiment_results)
    passed = sum(1 for r in experiment_results if r.get('score') == 'pass')
    failed = total - passed

    print(f"总样本数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    if total > 0:
        print(f"通过率: {passed/total*100:.1f}%")

    # 显示每个样本的详细信息
    print("\n详细结果:")
    print("-" * 60)
    for i, result in enumerate(experiment_results, 1):
        print(f"\n样本 {i} (索引 {result.get('sample_index')}):")
        print(f"  问题: {result.get('question', '')[:100]}...")
        print(f"  来源: {result.get('source', '')}")
        print(f"  RAG 答案: {result.get('response', '')[:200]}...")
        print(f"  评分: {result.get('score', '')}")
        print(f"  日志文件: {result.get('log_file', '')}")

    # 保存实验结果
    experiment_results.save()
    csv_path = Path("evals/experiments") / f"{experiment_results.name}.csv"
    print(f"\n[OK] 实验结果已保存到: {csv_path.resolve()}")

    # 额外保存 JSON 格式的详细结果
    json_path = Path("evals/experiments") / f"{experiment_results.name}_detailed.json"
    detailed_results = []
    for result in experiment_results:
        detailed_results.append(dict(result))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"[OK] 详细结果已保存到: {json_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
