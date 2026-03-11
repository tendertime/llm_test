"""
提取multifieldqa_zh的前100条数据
"""
import json
import os

# 文件路径
input_file = "e:/code/rag_test/longbench/extracted/data/multifieldqa_zh.jsonl"
output_file = "e:/code/rag_test/longbench/multifieldqa_zh_top100.json"

print("正在读取multifieldqa_zh数据集...")

try:
    # 读取JSONL文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析数据
    data_lines = content.strip().split('\n')
    data = [json.loads(line) for line in data_lines if line.strip()]

    print(f"原始数据集大小: {len(data)}")

    # 提取前100条
    subset = data[:100]

    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"数据已保存到: {output_file}")
    print(f"数据条数: {len(subset)}")

    # 显示第一条数据的样例
    if subset:
        print("\n数据样例（第一条）:")
        print(json.dumps(subset[0], ensure_ascii=False, indent=2))

    print("\n完成!")

except Exception as e:
    print(f"操作失败: {e}")
    import traceback
    traceback.print_exc()
