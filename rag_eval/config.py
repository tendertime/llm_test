"""
配置文件：管理 LLM API 配置
"""
import os

# SiliconFlow DeepSeek 配置
SILICONFLOW_API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    ""  # 请在环境变量或.env文件中设置API密钥
)
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V3.1-Terminus"

# OpenAI 配置（可选）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"

# 当前使用的 LLM 提供商
CURRENT_LLM_PROVIDER = "siliconflow"  # 可选: "siliconflow", "openai", "anthropic", "google"


def get_openai_client():
    """获取配置好的 OpenAI 客户端"""
    from openai import OpenAI

    if CURRENT_LLM_PROVIDER == "siliconflow":
        return OpenAI(
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL
        )
    elif CURRENT_LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAI(api_key=OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported LLM provider: {CURRENT_LLM_PROVIDER}")


def get_model_name():
    """获取当前使用的模型名称"""
    if CURRENT_LLM_PROVIDER == "siliconflow":
        return SILICONFLOW_MODEL
    elif CURRENT_LLM_PROVIDER == "openai":
        return OPENAI_MODEL
    else:
        raise ValueError(f"Unsupported LLM provider: {CURRENT_LLM_PROVIDER}")
