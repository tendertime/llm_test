"""
将LongBench数据转换为RAGAS测试所需的DataFrame格式

RAGAS评估所需的数据格式：
- question: 问题
- contexts: 上下文列表
- ground_truth: 真实答案
- answer: 生成的答案（可选，通常在模型生成后添加）
"""
import pandas as pd
import json
import os
from pathlib import Path

def convert_longbench_to_ragas(data):
    """
    将LongBench数据转换为RAGAS格式
    
    Args:
        data: LongBench数据列表
        
    Returns:
        转换后的RAGAS格式数据
    """
    ragas_data = []
    
    for item in data:
        # 提取字段
        question = item.get('input', '')
        context = item.get('context', '')
        answers = item.get('answers', [])
        dataset_name = item.get('dataset', '')
        
        # RAGAS格式要求ground_truth是列表格式
        ground_truth = answers if isinstance(answers, list) else [answers]
        
        # contexts需要是列表格式
        contexts = [context] if context else []
        
        ragas_item = {
            'question': question,
            'contexts': contexts,
            'ground_truth': ground_truth,
            'answer': '',  # 待模型生成后填充
            'dataset': dataset_name  # 保留原始数据集信息
        }
        
        ragas_data.append(ragas_item)
    
    return ragas_data

def load_jsonl(file_path):
    """加载JSONL格式文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_json(file_path):
    """加载JSON格式文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_dataset(input_path, output_path, limit=None):
    """
    处理单个数据集并保存为CSV
    
    Args:
        input_path: 输入文件路径
        output_path: 输出CSV文件路径
        limit: 限制处理的数据条数
    """
    print(f"处理数据集: {input_path}")
    
    # 根据文件扩展名加载数据
    if input_path.endswith('.jsonl'):
        data = load_jsonl(input_path)
    elif input_path.endswith('.json'):
        data = load_json(input_path)
    else:
        raise ValueError(f"不支持的文件格式: {input_path}")
    
    print(f"  原始数据条数: {len(data)}")
    
    # 限制数据条数
    if limit:
        data = data[:limit]
        print(f"  限制后数据条数: {len(data)}")
    
    # 转换为RAGAS格式
    ragas_data = convert_longbench_to_ragas(data)
    
    # 创建DataFrame
    df = pd.DataFrame(ragas_data)
    
    # 保存为CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  已保存到: {output_path}")
    print(f"  DataFrame形状: {df.shape}")
    
    # 显示前几条数据的样例
    print("\n  数据样例:")
    print(df.head(2).to_string())
    
    return df

def main():
    """主函数"""
    # 定义输入输出目录
    input_dir = "e:/code/rag_test/longbench"
    output_dir = "e:/code/rag_test/longbench/ragas"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("LongBench数据转换为RAGAS格式")
    print("=" * 60)
    
    # 定义要转换的数据集
    datasets = [
        # 已处理的前100条
        {
            'input': os.path.join(input_dir, 'multifieldqa_zh_top100.json'),
            'output': os.path.join(output_dir, 'multifieldqa_zh_top100.csv'),
            'limit': None
        },
        # 从extracted目录加载原始数据
        {
            'input': os.path.join(input_dir, 'extracted/data/multifieldqa_zh.jsonl'),
            'output': os.path.join(output_dir, 'multifieldqa_zh.csv'),
            'limit': 100
        },
        {
            'input': os.path.join(input_dir, 'extracted/data/passage_count.jsonl'),
            'output': os.path.join(output_dir, 'passage_count.csv'),
            'limit': 100
        },
        {
            'input': os.path.join(input_dir, 'extracted/data/narrativeqa.jsonl'),
            'output': os.path.join(output_dir, 'narrativeqa.csv'),
            'limit': 50
        },
        {
            'input': os.path.join(input_dir, 'extracted/data/hotpotqa.jsonl'),
            'output': os.path.join(output_dir, 'hotpotqa.csv'),
            'limit': 50
        },
        {
            'input': os.path.join(input_dir, 'extracted/data/gov_report.jsonl'),
            'output': os.path.join(output_dir, 'gov_report.csv'),
            'limit': 50
        },
        {
            'input': os.path.join(input_dir, 'extracted/data/qmsum.jsonl'),
            'output': os.path.join(output_dir, 'qmsum.csv'),
            'limit': 50
        },
        {
            'input': os.path.join(input_dir, 'extracted/data/multifieldqa_en.jsonl'),
            'output': os.path.join(output_dir, 'multifieldqa_en.csv'),
            'limit': 50
        }
    ]
    
    # 处理每个数据集
    all_dataframes = {}
    for dataset in datasets:
        if os.path.exists(dataset['input']):
            try:
                df = process_dataset(
                    dataset['input'],
                    dataset['output'],
                    dataset['limit']
                )
                dataset_name = os.path.splitext(os.path.basename(dataset['output']))[0]
                all_dataframes[dataset_name] = df
                print("\n" + "-" * 60 + "\n")
            except Exception as e:
                print(f"  处理失败: {e}\n")
                import traceback
                traceback.print_exc()
                print("\n" + "-" * 60 + "\n")
        else:
            print(f"文件不存在，跳过: {dataset['input']}\n" + "-" * 60 + "\n")
    
    print("\n" + "=" * 60)
    print("所有数据集转换完成!")
    print("=" * 60)
    print(f"\n输出目录: {output_dir}")
    print("\n生成的文件:")
    for file in os.listdir(output_dir):
        if file.endswith('.csv'):
            print(f"  - {file}")
    
    # 创建一个合并的DataFrame（包含所有数据集）
    if all_dataframes:
        print("\n\n创建合并数据集...")
        combined_df = pd.concat(all_dataframes.values(), ignore_index=True)
        combined_output = os.path.join(output_dir, 'combined_all.csv')
        combined_df.to_csv(combined_output, index=False, encoding='utf-8-sig')
        print(f"合并数据集已保存到: {combined_output}")
        print(f"合并数据集总条数: {len(combined_df)}")
        
        # 按数据集统计
        print("\n各数据集数据量:")
        print(combined_df['dataset'].value_counts().to_string())
    
    return all_dataframes

if __name__ == "__main__":
    main()
