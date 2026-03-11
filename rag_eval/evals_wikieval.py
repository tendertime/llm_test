import asyncio
import os
from pathlib import Path

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
    """
    从本地或 Hugging Face 加载 WikiEval 数据集

    Args:
        wikieval_path: 本地数据集路径

    Returns:
        pandas DataFrame (兼容 HuggingFace Dataset 接口)
    """
    import pandas as pd

    wikieval_path = Path(wikieval_path)

    # 首先尝试从本地 parquet 文件加载
    if wikieval_path.exists():
        print(f"正在从本地加载 WikiEval 数据集: {wikieval_path}")
        try:
            # 查找 parquet 文件
            parquet_files = list(wikieval_path.glob("*.parquet"))
            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
                print(f"[OK] 成功从本地加载数据集，共 {len(df)} 个样本")
                return df
            else:
                print("[WARN] 未找到 parquet 文件")
        except Exception as e:
            print(f"[WARN] 本地加载失败: {e}")
            print("尝试从 Hugging Face 下载...")

    # 如果本地加载失败，从 Hugging Face 下载
    if not DATASETS_AVAILABLE:
        print("[ERROR] datasets 包未安装，请先安装:")
        print("pip install datasets")
        return None

    print("正在从 Hugging Face 下载 WikiEval 数据集...")
    try:
        dataset = load_dataset("explodinggradients/wikieval", split="train")
        print(f"[OK] 成功加载 WikiEval 数据集，共 {len(dataset)} 个样本")
        return dataset
    except Exception as e:
        print(f"[ERROR] 加载数据集失败: {e}")
        return None


def prepare_rag_with_wikieval(wikieval_dataset, sample_size=5):
    """
    使用 WikiEval 数据准备 RAG 系统

    Args:
        wikieval_dataset: WikiEval 数据集（pandas DataFrame 或 HuggingFace Dataset）
        sample_size: 用于文档库的样本数量（默认 5 个）

    Returns:
        配置好的 RAG 客户端
    """
    print(f"\n正在准备 RAG 系统（使用前 {sample_size} 个样本的 context_v1 作为文档库）...")

    # 提取前 N 个样本的 context_v1 作为文档库
    documents = []
    for i in range(min(sample_size, len(wikieval_dataset))):
        # 支持 pandas DataFrame 和 HuggingFace Dataset
        if hasattr(wikieval_dataset, 'iloc'):
            # pandas DataFrame
            context_v1 = wikieval_dataset.iloc[i]['context_v1']
        else:
            # HuggingFace Dataset
            context_v1 = wikieval_dataset[i]['context_v1']

        if context_v1:  # 确保 context_v1 不为空
            documents.extend(context_v1)  # context_v1 是一个列表

    print(f"[OK] 文档库包含 {len(documents)} 个文档片段")

    # 创建 RAG 系统
    rag_client = ExampleRAG(
        llm_client=openai_client,
        retriever=SimpleKeywordRetriever(),
        logdir="evals/wikieval_logs"
    )

    # 添加文档到知识库
    rag_client.add_documents(documents)

    return rag_client


def create_evaluation_dataset(wikieval_dataset, sample_size=3):
    """
    创建用于评估的数据集

    Args:
        wikieval_dataset: WikiEval 数据集（pandas DataFrame 或 HuggingFace Dataset）
        sample_size: 用于评估的样本数量

    Returns:
        Ragas Dataset 对象
    """
    print(f"\n正在创建评估数据集（使用 {sample_size} 个测试样本）...")

    dataset = Dataset(
        name="wikieval_eval",
        backend="local/csv",
        root_dir="evals",
    )

    # 从数据集中选择样本（跳过用于文档库的样本）
    start_idx = 5  # 跳过前 5 个样本（用作文档库）
    end_idx = min(start_idx + sample_size, len(wikieval_dataset))

    for i in range(start_idx, end_idx):
        # 支持 pandas DataFrame 和 HuggingFace Dataset
        if hasattr(wikieval_dataset, 'iloc'):
            # pandas DataFrame
            sample = wikieval_dataset.iloc[i]
        else:
            # HuggingFace Dataset
            sample = wikieval_dataset[i]

        row = {
            "question": sample['question'],
            "ground_truth": sample['answer'],  # 使用数据集中的标准答案
            "source": sample['source']
        }
        dataset.append(row)

    # 保存数据集
    dataset.save()
    print(f"[OK] 评估数据集已保存")
    
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
    """
    # 使用 RAG 系统生成答案
    response = rag_client.query(row["question"])
    
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
    """
    主函数：执行 WikiEval 评估
    """
    print("=" * 60)
    print("WikiEval 数据集 RAG 系统评估")
    print("=" * 60)
    
    # 1. 加载 WikiEval 数据集
    wikieval_dataset = load_wikieval_dataset()
    if wikieval_dataset is None:
        return
    
    # 2. 准备 RAG 系统
    global rag_client
    rag_client = prepare_rag_with_wikieval(wikieval_dataset, sample_size=5)
    
    # 3. 创建评估数据集
    eval_dataset = create_evaluation_dataset(wikieval_dataset, sample_size=3)
    
    # 4. 运行实验
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
    print(f"通过率: {passed/total*100:.1f}%")

    # 显示每个样本的详细信息
    print("\n详细结果:")
    print("-" * 60)
    for i, result in enumerate(experiment_results, 1):
        print(f"\n样本 {i}:")
        print(f"  问题: {result.get('question', '')}")
        print(f"  标准答案: {result.get('ground_truth', '')}")
        print(f"  RAG 答案: {result.get('response', '')}")
        print(f"  评分: {result.get('score', '')}")
        print(f"  日志文件: {result.get('log_file', '')}")
    
    # 保存实验结果
    experiment_results.save()
    csv_path = Path("evals/experiments") / f"{experiment_results.name}.csv"
    print(f"\n[OK] 实验结果已保存到: {csv_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
