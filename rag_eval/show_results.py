"""显示评估结果汇总"""
import json
import pandas as pd
from pathlib import Path

print("=" * 70)
print("WikiEval 评估结果汇总")
print("=" * 70)

# 1. 检查 CSV 结果文件
experiments_dir = Path("evals/experiments")
csv_files = list(experiments_dir.glob("*.csv"))

if csv_files:
    print(f"\n找到 {len(csv_files)} 个实验结果文件:")
    for csv_file in sorted(csv_files, reverse=True):
        print(f"\n文件: {csv_file.name}")
        print(f"大小: {csv_file.stat().st_size:,} bytes")

        try:
            df = pd.read_csv(csv_file)
            print(f"样本数: {len(df)}")

            # 显示评分统计
            if 'score' in df.columns:
                passed = (df['score'] == 'pass').sum()
                failed = (df['score'] == 'fail').sum()
                pass_rate = (passed / len(df) * 100) if len(df) > 0 else 0
                print(f"通过: {passed}, 失败: {failed}, 通过率: {pass_rate:.1f}%")

        except Exception as e:
            print(f"读取失败: {e}")

# 2. 检查 JSON 详细结果
json_files = list(experiments_dir.glob("*_detailed.json"))

if json_files:
    print(f"\n找到 {len(json_files)} 个详细结果文件:")
    for json_file in sorted(json_files, reverse=True):
        print(f"\n文件: {json_file.name}")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            print(f"样本数: {len(results)}")

            # 显示评分统计
            passed = sum(1 for r in results if r.get('score') == 'pass')
            failed = sum(1 for r in results if r.get('score') == 'fail')
            pass_rate = (passed / len(results) * 100) if len(results) > 0 else 0
            print(f"通过: {passed}, 失败: {failed}, 通过率: {pass_rate:.1f}%")

            # 显示每个样本的摘要
            print("\n样本详情:")
            for i, result in enumerate(results, 1):
                print(f"\n  {i}. {result.get('source', 'Unknown')}")
                print(f"     问题: {result.get('question', '')[:80]}...")
                print(f"     RAG 答案: {result.get('response', '')[:100]}...")
                print(f"     评分: {result.get('score', 'N/A')}")

        except Exception as e:
            print(f"读取失败: {e}")

# 3. 检查日志文件
log_dirs = [
    Path("evals/wikieval_logs"),
    Path("evals/wikieval_fixed_logs"),
]

for log_dir in log_dirs:
    if log_dir.exists():
        log_files = list(log_dir.glob("*.json"))
        if log_files:
            print(f"\n{log_dir.name} 日志:")
            print(f"找到 {len(log_files)} 个日志文件")

            # 显示最新的几个日志摘要
            for log_file in sorted(log_files, reverse=True)[:3]:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)

                    query = log_data.get('query', 'N/A')
                    result = log_data.get('result', {})
                    answer = result.get('answer', 'N/A')[:80]

                    print(f"\n  日志: {log_file.name}")
                    print(f"  问题: {query[:60]}...")
                    print(f"  答案: {answer}...")

                except Exception as e:
                    print(f"  读取失败: {e}")

print("\n" + "=" * 70)
print("结果汇总完成")
print("=" * 70)
