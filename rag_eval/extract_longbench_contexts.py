"""
从 LongBench 数据集中提取所有 contexts 并生成文档列表
"""
import ast
from pathlib import Path

import pandas as pd


def extract_all_contexts():
    """提取所有 longbench 数据集中的 contexts"""
    longbench_dir = Path(__file__).parent.parent / "longbench" / "ragas"

    # 获取所有 CSV 文件
    csv_files = list(longbench_dir.glob("*.csv"))
    print(f"找到 {len(csv_files)} 个 CSV 文件")

    all_documents = []
    dataset_stats = {}

    for csv_file in csv_files:
        print(f"\n处理: {csv_file.name}")

        try:
            # 读取 CSV
            df = pd.read_csv(csv_file, encoding='utf-8-sig')

            # 提取 contexts
            doc_count = 0
            for idx, row in df.iterrows():
                contexts_str = row.get('contexts', '[]')

                # 解析 contexts
                try:
                    contexts = ast.literal_eval(contexts_str) if isinstance(contexts_str, str) else contexts_str
                except:
                    contexts = []

                # 添加到总列表
                for ctx in contexts:
                    if isinstance(ctx, str) and ctx.strip():
                        all_documents.append(ctx)
                        doc_count += 1

            dataset_stats[csv_file.stem] = {
                'rows': len(df),
                'contexts': doc_count
            }
            print(f"  行数: {len(df)}, 提取的文档数: {doc_count}")

        except Exception as e:
            print(f"  错误: {e}")
            continue

    print(f"\n总计提取: {len(all_documents)} 个文档")
    print("\n各数据集统计:")
    for name, stats in dataset_stats.items():
        print(f"  {name}: {stats['rows']} 行, {stats['contexts']} 个文档")

    return all_documents, dataset_stats


def generate_python_list(documents, max_preview=3):
    """生成 Python 列表代码"""
    lines = ["DOCUMENTS = ["]

    for i, doc in enumerate(documents):
        # 转义引号和换行符
        escaped_doc = doc.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

        # 预览模式：只显示前几个
        if max_preview > 0 and i < max_preview:
            preview = escaped_doc[:100] + "..." if len(escaped_doc) > 100 else escaped_doc
            lines.append(f'    "{preview}",')
        elif max_preview > 0:
            # 只显示数量，不显示内容
            lines.append(f'    # ... 还有 {len(documents) - max_preview} 个文档 ...')
            break
        else:
            lines.append(f'    "{escaped_doc}",')

    lines.append("]")
    return '\n'.join(lines)


if __name__ == "__main__":
    # 提取所有 contexts
    documents, stats = extract_all_contexts()

    # 保存到文件
    output_file = Path(__file__).parent / "longbench_documents.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc + '\n\n---DOC_SEPARATOR---\n\n')

    print(f"\n所有文档已保存到: {output_file}")
    print(f"总计: {len(documents)} 个文档")

    # 生成 Python 代码预览
    print("\n生成的 Python 代码预览（前3个文档）:")
    print(generate_python_list(documents, max_preview=3))
