"""
下载LongBench的完整数据包
"""
import json
import os
import requests
import zipfile
from pathlib import Path

# 设置输出目录
output_dir = "e:/code/rag_test/longbench"
os.makedirs(output_dir, exist_ok=True)

# Hugging Face上的数据包URL
zip_url = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
zip_path = os.path.join(output_dir, "data.zip")

print(f"正在下载LongBench数据包...")
print(f"URL: {zip_url}")

try:
    # 下载数据包
    response = requests.get(zip_url, timeout=120, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0

    print(f"文件大小: {total_size / (1024 * 1024):.2f} MB")

    # 保存文件
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                if downloaded_size % (1024 * 1024) == 0:  # 每MB显示一次进度
                    progress = (downloaded_size / total_size * 100) if total_size > 0 else 0
                    print(f"下载进度: {progress:.1f}%")

    print(f"[OK] 数据包已下载到: {zip_path}")

    # 解压文件
    print("\n正在解压文件...")
    extract_dir = os.path.join(output_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"[OK] 文件已解压到: {extract_dir}")

    # 查找multidoc_qa_zh相关文件
    print("\n正在查找multidoc_qa_zh相关文件...")
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if 'multidoc' in file.lower():
                print(f"  找到: {os.path.join(root, file)}")

                # 如果是multidoc_qa_zh，提取前100条
                if 'multidoc_qa_zh' in file.lower():
                    print(f"\n处理文件: {file}")
                    file_path = os.path.join(root, file)

                    # 读取数据
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 解析JSONL或JSON
                    if file.endswith('.jsonl'):
                        data_lines = content.strip().split('\n')
                        data = [json.loads(line) for line in data_lines if line.strip()]
                    else:
                        data = json.loads(content)
                        if isinstance(data, list):
                            pass  # 已经是列表
                        elif isinstance(data, dict):
                            data = list(data.values())

                    print(f"  原始数据条数: {len(data)}")

                    # 提取前100条
                    subset = data[:100]

                    # 保存为JSON文件
                    output_file = os.path.join(output_dir, "multidoc_qa_zh_top100.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(subset, f, ensure_ascii=False, indent=2)

                    print(f"[OK] 已提取前100条数据到: {output_file}")

                    # 显示第一条数据的样例
                    if subset:
                        print("\n数据样例（第一条）:")
                        print(json.dumps(subset[0], ensure_ascii=False, indent=2)[:500] + "...")

except Exception as e:
    print(f"[FAIL] 操作失败: {e}")
    import traceback
    traceback.print_exc()
