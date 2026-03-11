"""
测试 LLM 配置是否正确
"""
from openai import OpenAI
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, SILICONFLOW_MODEL

print("=" * 60)
print("测试 SiliconFlow DeepSeek 配置")
print("=" * 60)

print(f"\n1. 配置信息:")
print(f"   API Key: {SILICONFLOW_API_KEY[:10]}...{SILICONFLOW_API_KEY[-10:]}")
print(f"   Base URL: {SILICONFLOW_BASE_URL}")
print(f"   Model: {SILICONFLOW_MODEL}")

print(f"\n2. 创建客户端...")
client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL
)
print("   [OK] 客户端创建成功")

print(f"\n3. 测试简单请求...")
try:
    response = client.chat.completions.create(
        model=SILICONFLOW_MODEL,
        messages=[
            {"role": "user", "content": "Hello! Please say 'Configuration test successful!' in English."}
        ],
        max_tokens=50
    )

    result = response.choices[0].message.content.strip()
    print(f"   [OK] 请求成功")
    print(f"   响应: {result}")
    print(f"   Token 使用: {response.usage.total_tokens} tokens")

    print(f"\n4. 配置验证结果: [SUCCESS]")
    print(f"\n现在可以运行评估:")
    print(f"   python evals_wikieval.py")

except Exception as e:
    print(f"   [ERROR] 请求失败: {e}")
    print(f"\n配置验证结果: [FAILED]")
    print(f"\n请检查:")
    print(f"   1. API Key 是否正确")
    print(f"   2. 网络连接是否正常")
    print(f"   3. 模型名称是否正确")
    exit(1)

print("=" * 60)
