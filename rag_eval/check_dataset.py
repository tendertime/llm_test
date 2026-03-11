"""检查本地 WikiEval 数据集"""
from pathlib import Path

try:
    from datasets import load_from_disk
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

print("=" * 60)
print("检查本地 WikiEval 数据集")
print("=" * 60)

# 检查数据目录
data_path = Path("e:/code/rag_test/WikiEval/data")
print(f"\n数据目录: {data_path}")

if not data_path.exists():
    print("X 数据目录不存在")
    exit(1)

# 列出文件
files = list(data_path.rglob("*"))
print(f"\n找到 {len(files)} 个文件:")
for f in files:
    if f.is_file():
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")

# 尝试加载数据集
if DATASETS_AVAILABLE:
    print("\n正在加载数据集...")
    try:
        dataset = load_from_disk(str(data_path))
        print("v 数据集加载成功")
        
        print(f"\n数据集信息:")
        print(f"  - Splits: {list(dataset.keys())}")
        
        if 'train' in dataset:
            train = dataset['train']
            print(f"  - Train 样本数: {len(train)}")
            print(f"  - 特征字段: {list(train.features.keys())}")
            
            # 显示前 2 个样本
            print(f"\n前 2 个样本预览:")
            for i in range(min(2, len(train))):
                sample = train[i]
                print(f"\n样本 {i+1}:")
                print(f"  问题: {sample['question'][:80]}...")
                print(f"  答案: {sample['answer'][:80]}...")
                print(f"  来源: {sample['source']}")
                print(f"  context_v1 数量: {len(sample['context_v1'])}")
                print(f"  context_v2 数量: {len(sample['context_v2'])}")
                
        print("\nv 本地数据集可用，可以直接使用！")
        
    except Exception as e:
        print(f"X 加载失败: {e}")
        exit(1)
else:
    print("X datasets 包未安装")
    print("请运行: pip install datasets")
    exit(1)
