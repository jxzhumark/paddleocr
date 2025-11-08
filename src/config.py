import os
from pathlib import Path
from dotenv import load_dotenv

# 允许 .env 覆盖终端环境：你也可以改成 override=True
load_dotenv(override=False)

BASE = Path(__file__).resolve().parents[1]  # repo root

# 统一可复用的目录
DIR_CONFIG = BASE / "config"
DIR_DATA   = BASE / "data"
DIR_OUT    = BASE / "out"
DIR_MODEL  = BASE / "model"
DIR_RUN    = DIR_OUT / "run"     # 进程 pid/log 放这里
DIR_HF     = DIR_MODEL / "hf"    # vLLM 下载缓存（HF）
DIR_MS     = DIR_MODEL / "ms"    # vLLM 下载缓存（ModelScope）

# 两个 venv
VENV_PADDLE = BASE / "venvs" / "paddleocr" / "bin"
VENV_VLLM   = BASE / "venvs" / "vllm" / "bin"

def getenv(key: str, default=None):
    return os.getenv(key, default)
