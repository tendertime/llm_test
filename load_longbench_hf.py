"""
从Hugging Face加载LongBench数据集
"""
import json
import os
from huggingface_hub import hf_hub_download

# 设置输出目录
output_dir = "e:/code/rag_test/longbench"
os.makedirs(output_dir, exist_ok=True)

print("正在从Hugging Face下载LongBench数据集...")

try:
    # 方式1: 直接下载data.zip文件
    print("尝试下载完整数据包...")
    data_zip_path = hf_hub_download(
        repo_id="THUDM/LongBench",
        filename="data.zip",
        repo_type="dataset",
        local_dir=output_dir
    )
    print(f"数据包已下载到: {data_zip_path}")
    print("请手动解压data.zip文件")

except Exception as e:
    print(f"下载失败: {e}")
    print("\n尝试直接下载数据集...")

    try:
        # 尝试不同的数据集名称
        possible_datasets = [
            "multidoc_qa_zh",
            "multifieldqa_zh",
            "narrativeqa",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "gov_report",
            "qmsum",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "multifieldqa_en",
            "qasper",
            "dureader",
            "vcsum",
            "lsht",
            "passage_retrieval_zh",
            "lcc",
            "repobench-p"
        ]

        for dataset_name in possible_datasets:
            try:
                print(f"\n尝试下载 {dataset_name}...")
                file_path = hf_hub_download(
                    repo_id="THUDM/LongBench",
                    filename=f"data/{dataset_name}.jsonl",
                    repo_type="dataset",
                    local_dir=output_dir
                )
                print(f"成功下载: {file_path}")
            except Exception as dataset_e:
                print(f"  {dataset_name} 下载失败: {dataset_e}")
                continue

    except Exception as e2:
        print(f"直接下载也失败: {e2}")
