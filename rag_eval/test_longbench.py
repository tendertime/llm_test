"""
使用 LongBench 数据集测试 RAG 系统
"""
import ast
import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))
from rag import default_rag_client

# Configuration
DATASET_NAME = "multifieldqa_zh"  # 可选: multifieldqa_zh, multifieldqa_en, qmsum, etc.
SAMPLE_SIZE = 5  # 测试样本数量
LOG_DIR = "evals/longbench_logs"


def load_longbench_data(dataset_name: str = "multifieldqa_zh", sample_size: int = 5):
    """
    加载 LongBench 数据集

    Args:
        dataset_name: 数据集名称（不含.csv后缀）
        sample_size: 要加载的样本数量

    Returns:
        pandas DataFrame
    """
    data_path = Path(__file__).parent.parent / "longbench" / "ragas" / f"{dataset_name}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {data_path}")

    # 读取 CSV（使用 utf-8-sig 编码处理 BOM）
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    # 只取前 sample_size 个样本
    df = df.head(sample_size)

    print(f"[OK] 加载了 {len(df)} 个样本 from {dataset_name}")
    print(f"数据列: {df.columns.tolist()}")

    return df


def prepare_contexts(row):
    """
    准备上下文数据

    Args:
        row: 数据行

    Returns:
        contexts 列表
    """
    contexts = row['contexts']
    if isinstance(contexts, str):
        try:
            # 将字符串形式的列表转换为实际列表
            contexts = ast.literal_eval(contexts)
        except:
            contexts = [contexts]
    elif not isinstance(contexts, list):
        contexts = [contexts]

    return contexts


# Initialize SiliconFlow client
api_key = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-cxwvirgzjrvlvleqxzwobedlbcetqrgtqyhsydylujozahnf"
)
openai_client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)

# Initialize RAG client and LLM for evaluation
# 自动加载 LongBench 知识库
rag_client = default_rag_client(
    llm_client=openai_client,
    logdir=LOG_DIR,
    model_name="deepseek-ai/DeepSeek-V3.1-Terminus",
    embedding_model="BAAI/bge-large-zh-v1.5",  # SiliconFlow embeddings 模型
    use_longbench=True,  # 自动加载 LongBench 知识库
    longbench_datasets=None  # None = 加载所有，或指定 ['multifieldqa_zh', 'qmsum']
)

llm = llm_factory("deepseek-ai/DeepSeek-V3.1-Terminus", client=openai_client)


def create_ragas_dataset(df: pd.DataFrame):
    """
    创建 RAGAS Dataset

    Args:
        df: LongBench 数据 DataFrame

    Returns:
        RAGAS Dataset
    """
    dataset = Dataset(
        name=f"longbench_{DATASET_NAME}",
        backend="local/csv",
        root_dir="evals",
    )

    for idx, row in df.iterrows():
        contexts = prepare_contexts(row)
        ground_truth = row.get('ground_truth', '')

        # 处理 ground_truth 格式
        if isinstance(ground_truth, str):
            try:
                ground_truth = ast.literal_eval(ground_truth)
                if isinstance(ground_truth, list):
                    ground_truth = ground_truth[0] if ground_truth else ""
            except:
                pass

        dataset_row = {
            "question": row['question'],
            "contexts": str(contexts),  # 转为字符串存储
            "ground_truth": str(ground_truth),
            "dataset": row.get('dataset', DATASET_NAME),
        }
        dataset.append(dataset_row)

    dataset.save()
    print(f"[OK] 创建了 RAGAS Dataset: {dataset.name}")
    return dataset


# 定义评估指标
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt=(
        "检查回答是否准确回答了问题。"
        "问题: {question}\n"
        "回答: {response}\n"
        "参考答案: {ground_truth}\n"
        "如果回答准确且包含关键信息，返回 'pass'，否则返回 'fail'。"
    ),
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_longbench_experiment(row):
    """
    运行 LongBench 实验

    Args:
        row: 数据行

    Returns:
        实验结果字典
    """
    # 执行查询（使用全局 LongBench 知识库）
    question = row['question']
    response = rag_client.query(question, top_k=3)

    # 评估答案质量
    ground_truth = row.get('ground_truth', '')
    score = correctness_metric.score(
        llm=llm,
        question=question,
        response=response.get('answer', ''),
        ground_truth=ground_truth,
    )

    # 构建实验结果
    experiment_result = {
        **row,
        "rag_answer": response.get('answer', ''),
        "score": score.value,
        "log_file": response.get('logs', ''),
    }

    return experiment_result


async def main():
    """主函数"""
    print("=" * 70)
    print(f"LongBench RAG 测试 - 数据集: {DATASET_NAME}")
    print("=" * 70)

    # 1. 加载数据
    print(f"\n[1/4] 加载 {DATASET_NAME} 数据集...")
    df = load_longbench_data(DATASET_NAME, SAMPLE_SIZE)

    # 2. 创建 RAGAS Dataset
    print(f"\n[2/4] 创建 RAGAS Dataset...")
    dataset = create_ragas_dataset(df)

    # 3. 运行实验
    print(f"\n[3/4] 运行 RAG 实验...")
    print(f"处理 {len(df)} 个样本...")

    experiment_results = await run_longbench_experiment.arun(dataset)

    print("\n[OK] 实验完成!")

    # 4. 保存结果
    print(f"\n[4/4] 保存实验结果...")
    experiment_results.save()

    # 显示结果摘要
    print("\n" + "=" * 70)
    print("实验结果摘要")
    print("=" * 70)

    # 统计通过率
    results_df = experiment_results.to_pandas()
    if 'score' in results_df.columns:
        pass_count = (results_df['score'] == 'pass').sum()
        total_count = len(results_df)
        pass_rate = pass_count / total_count * 100 if total_count > 0 else 0
        print(f"\n通过率: {pass_count}/{total_count} ({pass_rate:.1f}%)")

    # 保存详细结果
    csv_path = Path(".") / "experiments" / f"{experiment_results.name}.csv"
    print(f"\n详细结果已保存到: {csv_path.resolve()}")

    # 显示前几个样本的结果
    print("\n前几个样本结果:")
    print("-" * 70)
    for idx, row in results_df.head(3).iterrows():
        print(f"\n样本 {idx + 1}:")
        print(f"问题: {row.get('question', '')[:100]}...")
        print(f"RAG答案: {row.get('rag_answer', '')[:150]}...")
        print(f"得分: {row.get('score', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
