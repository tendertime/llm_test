"""
下载和准备 WikiEval 数据集
"""
import os
from pathlib import Path

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


def download_wikieval():
    """
    从 Hugging Face 下载 WikiEval 数据集并保存到本地
    """
    if not DATASETS_AVAILABLE:
        print("✗ datasets 包未安装")
        print("请运行: uv add datasets")
        return False
    
    print("=" * 60)
    print("下载 WikiEval 数据集")
    print("=" * 60)
    
    # 定义保存路径
    save_path = Path("e:/code/rag_test/WikiEval/data")
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n正在从 Hugging Face 下载 WikiEval 数据集...")
    print(f"保存路径: {save_path}")
    
    try:
        # 下载数据集
        dataset = load_dataset("explodinggradients/wikieval")
        
        # 保存到本地
        dataset.save_to_disk(str(save_path))
        
        print(f"\n✓ 数据集下载成功！")
        print(f"  - 训练集样本数: {len(dataset['train'])}")
        print(f"  - 保存路径: {save_path}")
        
        # 显示数据集信息
        print(f"\n📊 数据集字段:")
        for field in dataset['train'].features:
            print(f"  - {field}")
        
        # 显示前 3 个样本
        print(f"\n📝 前 3 个样本预览:")
        for i in range(min(3, len(dataset['train']))):
            print(f"\n样本 {i+1}:")
            sample = dataset['train'][i]
            print(f"  问题: {sample['question'][:100]}...")
            print(f"  答案: {sample['answer'][:100]}...")
            print(f"  来源: {sample['source']}")
            print(f"  context_v1 数量: {len(sample['context_v1'])}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False


def check_local_dataset():
    """
    检查本地是否有 WikiEval 数据集
    """
    data_path = Path("e:/code/rag_test/WikiEval/data")
    
    print("=" * 60)
    print("检查本地 WikiEval 数据集")
    print("=" * 60)
    
    if not data_path.exists():
        print(f"✗ 数据目录不存在: {data_path}")
        return False
    
    # 检查是否有数据文件
    files = list(data_path.rglob("*"))
    if not files:
        print(f"✗ 数据目录为空: {data_path}")
        return False
    
    print(f"✓ 数据目录存在: {data_path}")
    print(f"  找到 {len(files)} 个文件")
    
    # 尝试加载数据集
    if DATASETS_AVAILABLE:
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(str(data_path))
            print(f"  - 可以加载的数据集: {list(dataset.keys())}")
            if 'train' in dataset:
                print(f"  - 训练集样本数: {len(dataset['train'])}")
            return True
        except Exception as e:
            print(f"⚠ 数据集加载失败: {e}")
            print("可能需要重新下载数据集")
            return False
    else:
        print("⚠ datasets 包未安装，无法验证数据集内容")
        return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # 检查本地数据集
        check_local_dataset()
    else:
        # 下载数据集
        success = download_wikieval()
        
        if success:
            print("\n" + "=" * 60)
            print("下一步：")
            print("=" * 60)
            print("1. 查看数据集内容:")
            print("   python prepare_wikieval.py check")
            print("\n2. 运行评估:")
            print("   uv run python evals_wikieval.py")
