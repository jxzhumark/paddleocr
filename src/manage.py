#!/usr/bin/env python3
import argparse, os, sys, time, json, signal
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from subprocess import Popen, DEVNULL, STDOUT

from src.config import (
    BASE, DIR_CONFIG, DIR_DATA, DIR_OUT, DIR_RUN, DIR_MODEL, DIR_HF, DIR_MS,
    VENV_PADDLE, VENV_VLLM
)

# ----------------------------
# 端口
# ----------------------------
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8118                # vLLM OpenAI 兼容服务
PIPE_HOST = "127.0.0.1"
PIPE_PORT = 8080                # PaddleX 产线 /layout-parsing

# ----------------------------
# 重要文件
# ----------------------------
PID_VLLM   = DIR_RUN / "vllm.pid"
LOG_VLLM   = DIR_RUN / "vllm.log"
PID_PIPE   = DIR_RUN / "paddlex.pid"
LOG_PIPE   = DIR_RUN / "paddlex.log"

PIPELINE_CFG = DIR_CONFIG / "pipeline_config.yaml"

def ensure_dirs():
    for d in [DIR_CONFIG, DIR_DATA, DIR_OUT, DIR_RUN, DIR_MODEL, DIR_HF, DIR_MS]:
        d.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 进程工具
# ----------------------------
def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # 仅检测
        return True
    except OSError:
        return False

def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None

def _kill(pid_file: Path, name: str, timeout=10):
    pid = _read_pid(pid_file)
    if not pid:
        print(f"[=] {name}: no pid")
        return
    if not _is_alive(pid):
        print(f"[=] {name}: not running (stale pid {pid})")
        try: pid_file.unlink()
        except Exception: pass
        return
    print(f"[~] stopping {name} (pid={pid}) ...")
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not _is_alive(pid):
            break
        time.sleep(0.3)
    if _is_alive(pid):
        print(f"[!] {name}: SIGKILL")
        try: os.kill(pid, signal.SIGKILL)
        except Exception: pass
    try: pid_file.unlink()
    except Exception: pass
    print(f"[✓] {name} stopped")

def _tail(path: Path, n: int = 200):
    if not path.exists():
        print(f"[!] log not found: {path}")
        return
    with open(path, "rb") as f:
        try:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 8192
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
        except Exception:
            f.seek(0)
            data = f.read()
    sys.stdout.write(data.decode(errors="ignore"))

# ----------------------------
# HTTP 小工具 & 按页自检
# ----------------------------
def wait_http(url, timeout_s=60):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            req = Request(url, headers={"User-Agent": "curl/7.79"})
            with urlopen(req, timeout=5) as r:
                return 200 <= r.status < 500
        except (URLError, HTTPError):
            time.sleep(1.5)
    return False

def _b64_of_file(p: Path) -> str:
    import base64
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def check_layout_parsing_pages(api_url: str, pdf_path: Path, expect_pages: int,
                               timeout_s: float = 180.0,
                               visualize: bool = False,
                               prettify_md: bool = False) -> tuple[bool, str]:
    """POST /layout-parsing，自检返回页数是否达到阈值。"""
    t0 = time.time()
    payload = {
        "file": _b64_of_file(pdf_path),
        "fileType": 0,  # 0=PDF
        "visualize": bool(visualize),
        "prettifyMarkdown": bool(prettify_md),
        "vl_rec_max_concurrency": 1,
        "maxPixels": 1_500_000,
    }
    headers = {"Content-Type": "application/json"}
    last_err = ""
    while time.time() - t0 < timeout_s:
        try:
            r = urlopen(Request(api_url, data=json.dumps(payload).encode(), headers=headers), timeout=10)
            if 200 <= r.status < 300:
                resp = json.loads(r.read().decode())
                result = resp.get("result") or {}
                pages = result.get("layoutParsingResults") or []
                ok_pages = sum(1 for p in pages if isinstance(p.get("markdown", {}), dict))
                if len(pages) >= expect_pages and ok_pages >= expect_pages:
                    return True, f"got {len(pages)} (markdown pages={ok_pages})"
                last_err = f"pages={len(pages)}, md_pages={ok_pages} < expected={expect_pages}"
            else:
                last_err = f"http {r.status}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2)
    return False, last_err or "timeout"

# ----------------------------
# 启动 vLLM & PaddleX
# ----------------------------
def start_vllm():
    """venvs/vllm 中用 vllm serve 起 OpenAI 兼容服务。"""
    ensure_dirs()
    exe = VENV_VLLM / "vllm"
    if not exe.exists():
        raise RuntimeError(f"vLLM not found: {exe}; did you create venvs/vllm ?")

    env = os.environ.copy()
    env["HF_HOME"] = str(DIR_HF)
    env["HUGGINGFACE_HUB_CACHE"] = str(DIR_HF)
    env["MODELSCOPE_CACHE"] = str(DIR_MS)
    env["VLLM_USE_MODELSCOPE"] = "True"   # 允许走 ModelScope 镜像
    cmd = [
        str(exe), "serve", "PaddlePaddle/PaddleOCR-VL",
        "--trust-remote-code",
        "--download-dir", str(DIR_HF),
        "--max-num-batched-tokens", "16384",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--host", VLLM_HOST, "--port", str(VLLM_PORT),
    ]
    LOG_VLLM.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOG_VLLM, "ab", buffering=0)
    proc = Popen(cmd, cwd=str(BASE), stdout=f, stderr=STDOUT, stdin=DEVNULL, env=env)
    PID_VLLM.write_text(str(proc.pid))
    print(f"[+] vLLM started pid={proc.pid} → http://{VLLM_HOST}:{VLLM_PORT}/v1")

def start_paddlex():
    """venvs/paddleocr 中用 paddlex --serve 起 /layout-parsing。"""
    ensure_dirs()
    exe = VENV_PADDLE / "paddlex"
    if not exe.exists():
        raise RuntimeError(f"paddlex not found: {exe}; did you create venvs/paddleocr ?")

    env = os.environ.copy()
    # 把 Paddlex / Paddle 缓存指到 model/ 之下（可选）
    env["PADDLEX_HOME"] = str(DIR_MODEL / ".paddlex")
    env["PADDLEX_CACHE_DIR"] = str(DIR_MODEL / ".paddlex" / "temp")
    env["PADDLE_HUB_HOME"] = str(DIR_MODEL / ".paddlex" / "official_models")
    env["PPNLP_HOME"] = str(DIR_MODEL / ".paddlenlp")
    env["HF_HOME"] = env.get("HF_HOME", str(DIR_HF))
    env["HUGGINGFACE_HUB_CACHE"] = env.get("HUGGINGFACE_HUB_CACHE", str(DIR_HF))

    cmd = [
        str(exe), "--serve",
        "--pipeline", str(PIPELINE_CFG),
        "--host", PIPE_HOST, "--port", str(PIPE_PORT),
        "--device", "gpu",
    ]
    LOG_PIPE.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOG_PIPE, "ab", buffering=0)
    proc = Popen(cmd, cwd=str(BASE), stdout=f, stderr=STDOUT, stdin=DEVNULL, env=env)
    PID_PIPE.write_text(str(proc.pid))
    print(f"[+] PaddleX started pid={proc.pid} → http://{PIPE_HOST}:{PIPE_PORT}/docs")

# ----------------------------
# 子命令
# ----------------------------
def up(args):
    ensure_dirs()

    # 起 vLLM
    start_vllm()
    # 等 vLLM 就绪（/v1/models）
    ok = wait_http(f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models", 180)
    print(f"[{'OK' if ok else '!!'}] vLLM /v1/models")

    # 起 PaddleX
    start_paddlex()
    ok2 = wait_http(f"http://{PIPE_HOST}:{PIPE_PORT}/docs", 180)
    print(f"[{'OK' if ok2 else '!!'}] PaddleX /docs")

    # 按页自检
    api_layout = f"http://{PIPE_HOST}:{PIPE_PORT}/layout-parsing"
    one = DIR_DATA / "paf" / "one-page.pdf"
    multi = DIR_DATA / "paf" / "multi-page.pdf"

    if one.exists():
        ok_p1, msg1 = check_layout_parsing_pages(api_layout, one, expect_pages=1, timeout_s=180)
        print(f"[{'OK' if ok_p1 else '!!'}] layout-parsing (1p): {msg1}")
    else:
        print("[~] skip: data/paf/one-page.pdf not found")
    if multi.exists():
        ok_pm, msgm = check_layout_parsing_pages(api_layout, multi, expect_pages=2, timeout_s=240)
        print(f"[{'OK' if ok_pm else '!!'}] layout-parsing (multi): {msgm}")
    else:
        print("[~] skip: data/paf/multi-page.pdf not found")

    print("[✓] All services started.")

def down(_):
    _kill(PID_PIPE, "paddlex")
    _kill(PID_VLLM, "vllm")
    print("[✓] all stopped")

def status(_):
    for name, pid_file in [("vllm", PID_VLLM), ("paddlex", PID_PIPE)]:
        pid = _read_pid(pid_file)
        if pid and _is_alive(pid):
            print(f"{name:8} RUNNING pid={pid}")
        elif pid:
            print(f"{name:8} DEAD    (stale pid={pid})")
        else:
            print(f"{name:8} <not started>")

def logs(args):
    if args.name == "vllm":
        _tail(LOG_VLLM, n=args.tail)
    elif args.name == "paddlex":
        _tail(LOG_PIPE, n=args.tail)

def hc(_):
    print("[~] healthcheck...")
    ok = wait_http(f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models", 30)
    print(f"[{'OK' if ok else '!!'}] vLLM /v1/models")
    ok2 = wait_http(f"http://{PIPE_HOST}:{PIPE_PORT}/docs", 30)
    print(f"[{'OK' if ok2 else '!!'}] PaddleX /docs")

    api_layout = f"http://{PIPE_HOST}:{PIPE_PORT}/layout-parsing"
    one = DIR_DATA / "paf" / "one-page.pdf"
    multi = DIR_DATA / "paf" / "multi-page.pdf"
    if one.exists():
        ok_p1, msg1 = check_layout_parsing_pages(api_layout, one, expect_pages=1, timeout_s=120)
        print(f"[{'OK' if ok_p1 else '!!'}] layout-parsing (1p): {msg1}")
    if multi.exists():
        ok_pm, msgm = check_layout_parsing_pages(api_layout, multi, expect_pages=2, timeout_s=180)
        print(f"[{'OK' if ok_pm else '!!'}] layout-parsing (multi): {msgm}")

def main():
    p = argparse.ArgumentParser(description="PaddleOCR-VL local process manager")
    sub = p.add_subparsers(required=True)

    p_up = sub.add_parser("up", help="start vLLM + PaddleX")
    p_up.set_defaults(func=up)

    p_down = sub.add_parser("down", help="stop all")
    p_down.set_defaults(func=down)

    p_stat = sub.add_parser("status", help="show status")
    p_stat.set_defaults(func=status)

    p_logs = sub.add_parser("logs", help="tail logs")
    p_logs.add_argument("name", choices=["vllm", "paddlex"])
    p_logs.add_argument("--tail", type=int, default=200)
    p_logs.set_defaults(func=logs)

    p_hc = sub.add_parser("hc", help="healthcheck")
    p_hc.set_defaults(func=hc)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
