"""
RAGAS评估示例脚本

展示如何使用转换后的LongBench数据进行RAGAS评估
"""
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall
)
from datasets import Dataset
import json

def load_ragas_dataset(csv_path):
    """
    加载RAGAS格式的数据集
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        pandas DataFrame
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"加载数据集: {csv_path}")
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    return df

def prepare_for_ragas(df):
    """
    准备RAGAS评估所需的数据格式
    
    注意：answer列需要在模型生成后填充
    """
    # 确保contexts是列表格式
    if 'contexts' in df.columns:
        df['contexts'] = df['contexts'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    # 确保ground_truth是列表格式
    if 'ground_truth' in df.columns:
        df['ground_truth'] = df['ground_truth'].apply(lambda x: eval(x) if isinstance(x, str) else [x] if not isinstance(x, list) else x)
    
    return df

def convert_to_huggingface_dataset(df):
    """
    将DataFrame转换为HuggingFace Dataset格式
    
    Args:
        df: pandas DataFrame
        
    Returns:
        HuggingFace Dataset
    """
    # 转换为字典列表
    data = df.to_dict('records')
    
    # 创建Dataset
    dataset = Dataset.from_list(data)
    return dataset

def evaluate_with_ragas(dataset, metrics=None):
    """
    使用RAGAS进行评估
    
    Args:
        dataset: HuggingFace Dataset对象
        metrics: 要使用的评估指标列表
        
    Returns:
        评估结果
    """
    if metrics is None:
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    
    print(f"\n开始RAGAS评估...")
    print(f"使用的指标: {[m.name for m in metrics]}")
    print(f"评估样本数: {len(dataset)}")
    
    # 运行评估
    result = evaluate(
        dataset=dataset,
        metrics=metrics
    )
    
    return result

def sample_evaluation_demo():
    """
    示例：对少量数据进行评估演示
    """
    print("=" * 60)
    print("RAGAS评估示例")
    print("=" * 60)
    
    # 1. 加载数据
    csv_path = "e:/code/rag_test/longbench/ragas/multifieldqa_zh_top100.csv"
    df = load_ragas_dataset(csv_path)
    
    # 2. 准备数据
    df = prepare_for_ragas(df)
    
    # 3. 查看数据样例
    print("\n数据样例:")
    print(df[['question', 'ground_truth', 'answer']].head(2))
    
    # 4. 注意：需要先填充answer列
    print("\n" + "=" * 60)
    print("注意:")
    print("1. 'answer'列需要先用RAG系统生成答案")
    print("2. 生成答案后才能进行RAGAS评估")
    print("3. 以下是一个模拟示例")
    print("=" * 60)
    
    # 5. 模拟生成答案（实际使用时需要用RAG系统生成）
    # 这里只是演示，实际应该使用真实的RAG系统
    sample_df = df.head(5).copy()
    sample_df['answer'] = sample_df['ground_truth'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
    
    # 6. 转换为HuggingFace Dataset
    sample_dataset = convert_to_huggingface_dataset(sample_df)
    
    print(f"\n样本数据集大小: {len(sample_dataset)}")
    print("\n样本数据:")
    for i in range(min(2, len(sample_dataset))):
        print(f"\n样本 {i+1}:")
        print(f"  问题: {sample_dataset[i]['question']}")
        print(f"  答案: {sample_dataset[i]['answer']}")
        print(f"  真实答案: {sample_dataset[i]['ground_truth']}")

def full_evaluation_example():
    """
    完整评估流程示例（需要先生成答案）
    """
    print("\n" + "=" * 60)
    print("完整评估流程示例")
    print("=" * 60)
    
    # 步骤1: 加载数据
    csv_path = "e:/code/rag_test/longbench/ragas/multifieldqa_zh_top100.csv"
    df = load_ragas_dataset(csv_path)
    df = prepare_for_ragas(df)
    
    # 步骤2: 用RAG系统生成答案（需要实现）
    print("\n步骤2: 用RAG系统生成答案")
    print("  注意: 这里需要实现你的RAG系统")
    print("  示例代码:")
    print("""
    def generate_answers_with_rag(questions, contexts):
        # 使用你的RAG系统生成答案
        answers = []
        for question, context in zip(questions, contexts):
            # 调用你的RAG系统
            answer = your_rag_system.generate(question, context)
            answers.append(answer)
        return answers
    
    df['answer'] = generate_answers_with_rag(df['question'], df['contexts'])
    """)
    
    # 步骤3: 保存带答案的数据集
    output_with_answers = "e:/code/rag_test/longbench/ragas/multifieldqa_zh_top100_with_answers.csv"
    df.to_csv(output_with_answers, index=False, encoding='utf-8-sig')
    print(f"\n已保存带答案的数据集: {output_with_answers}")
    
    # 步骤4: 运行RAGAS评估
    print("\n步骤4: 运行RAGAS评估")
    print("  注意: 需要安装ragas库")
    print("  pip install ragas")
    print("""
    # 转换为HuggingFace Dataset
    dataset = convert_to_huggingface_dataset(df)
    
    # 运行评估
    result = evaluate_with_ragas(dataset)
    
    # 查看结果
    print(result)
    
    # 保存结果
    result.to_pandas().to_csv("evaluation_results.csv", index=False)
    """)

def show_available_datasets():
    """
    显示所有可用的数据集
    """
    print("\n" + "=" * 60)
    print("可用的LongBench数据集（RAGAS格式）")
    print("=" * 60)
    
    import os
    ragas_dir = "e:/code/rag_test/longbench/ragas"
    
    if os.path.exists(ragas_dir):
        files = os.listdir(ragas_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        
        print("\n生成的数据集:")
        for file in csv_files:
            file_path = os.path.join(ragas_dir, file)
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            print(f"\n  {file}:")
            print(f"    数据条数: {len(df)}")
            print(f"    列名: {', '.join(df.columns.tolist())}")
            
            # 显示数据集信息
            if 'dataset' in df.columns:
                print(f"    数据集类型: {df['dataset'].iloc[0]}")

def main():
    """主函数"""
    # 显示可用数据集
    show_available_datasets()
    
    # 运行示例评估
    sample_evaluation_demo()
    
    # 展示完整评估流程
    full_evaluation_example()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
