# PaddleOCR-VL（Single Host）

> 该项目可以直接在单机（如 RTX 4090）上运行 PaddleOCR-VL 模型，提供 PDF/图片的文档布局分析与内容识别，输出 Markdown 格式文本与可视化结果。使用 PaddleX 产线封装 vLLM OpenAI 兼容后端，统一对外提供 /layout-parsing 接口，从而实现 PaddleX 与 vllm-server 的优势集成。

> PaddleX 产线能提供更丰富的预处理与后处理功能，如 PDF 分块、逐页输出 Markdown 等，极大提升长文档处理的效率与效果，同时简化用户调用流程。

> 另一方面，vLLM 后端可充分利用其高效的批处理与并发能力，提升模型推理的吞吐量与响应速度；同时保持与 OpenAI 接口的兼容性，方便集成到现有系统中。若用户已有 PaddleX 产线经验，则可无缝迁移至该一体化方案，享受更优的性能与易用性。此外，该方案支持灵活的模型管理与扩展，便于未来集成更多先进的视觉语言模型，如 PaddleOCR-VL 的升级版本或 DeepSeek-OCR 等。

> 最后，该项目是工业大模型的基础，一方面可用于数据标注与文档理解，另一方面可作为下游应用的基础组件，如 RAG 系统的文档解析组件，

**一体化文档解析服务**

1. 下层：vLLM 原生 OpenAI 兼容服务，直接 serve PaddleOCR-VL 模型
2. 上层：PaddleX 产线服务，对外提供统一的 /layout-parsing API
   > PaddleX 产线可将 VLM 后端切换为 vllm-server 并通过 server_url 指向 OpenAI 兼容接口。参见官方产线文档示例。
   > vLLM 提供 OpenAI 兼容的 /v1/\* 服务端（Chat/Completions/Models 等）。

## 架构图 & 端口

```
Single Host (RTX 4090/other GPU)
 ├─ vLLM OpenAI 兼容服务 :8118 (自己选择端口)
 │   vllm serve 直接加载 PaddleOCR-VL
 │   --download-dir 指向 ./model/hf
 │   --served-model-name PaddleOCR-VL-0.9B (选择合适的模型名与PaddleX产线配置对应)
 └─ PaddleX 产线服务 :8080
     pipeline_config.yaml → http://127.0.0.1:8118/v1  (backend: vllm-server)
     对外只暴露 /layout-parsing
```

## 目录结构

```
paddleocr/
├─ config/
│  ├─ pipeline_config.yaml      # PaddleX 产线配置（指向 vLLM）
├─ data/
│  └─ paf/
│     ├─ one-page.pdf
│     └─ multi-page.pdf
├─ out/                         # 输出（json/md/可视化）
├─ venvs/
│  ├─ paddleocr/                # 主环境（Paddle + PaddleOCR + PaddleX）
│  └─ vllm/                     # vLLM 后端环境（OpenAI 兼容）
├─ model/
│  ├─ hf/                       # vLLM/HF 缓存
│  └─ ms/                       # （可选）ModelScope 缓存
└─ src/
   ├─ config.py
   ├─ pipeline.py               # 已支持 PDF 分块与逐页 Markdown
   └─ manage.py                 # 进程管理/健康检查（本地版）
├─ README.md
├─ runcheck.py                  # 长文档分块+自检脚本
```

## 先决条件

1. NVIDIA 驱动 & CUDA 对应 40 系显卡（RTX 4090）
2. Python 3.11/3.12（推荐使用 uv 管理虚拟环境）
3. PaddlePaddle ≥ 3.2.1 且安装特制版 safetensors（Paddle 官方明确要求）
   > 如需国内镜像：清华 TUNA PyPI 镜像 https://pypi.tuna.tsinghua.edu.cn/simple

## 环境与启动（两套 uv 虚拟环境）

> 所有模型/权重统一写入 ./model/。Hugging Face 与 ModelScope 的缓存变量见下文。

### A) vLLM 后端（OpenAI 兼容，端口 :8118）

1. 创建并进入环境
   ```
   uv venv venvs/vllm --python=3.12
   source venvs/vllm/bin/activate
   ```
2. 安装 vLLM（nightly，按官方建议）
   ```
   uv pip install -U vllm --pre \
     -i https://pypi.tuna.tsinghua.edu.cn/simple \
     --extra-index-url https://wheels.vllm.ai/nightly \
     --extra-index-url https://download.pytorch.org/whl/cu129 \
     --index-strategy unsafe-best-match
   ```
3. 设定缓存目录并启动服务（OpenAI 兼容 /v1/\*）
   ```
   export HF_HOME="$PWD/model/hf"
   export HUGGINGFACE_HUB_CACHE="$PWD/model/hf"
   export MODELSCOPE_CACHE="$PWD/model/ms"
   export VLLM_USE_MODELSCOPE=True
   vllm serve PaddlePaddle/PaddleOCR-VL \
   --served-model-name PaddleOCR-VL-0.9B \
   --trust-remote-code \
   --download-dir ./model/hf \
   --max-num-batched-tokens 16384 \
   --no-enable-prefix-caching \
   --mm-processor-cache-gb 0 \
   --host 0.0.0.0 \
   --port 8118
   ```

### B) PaddleX 产线（对外 :8080，统一 /layout-parsing）

1.  创建并进入环境
    ```
    uv venv venvs/paddleocr --python=3.11
    source venvs/paddleocr/bin/activate
    ```
2.  安装依赖（Paddle + PaddleOCR + PaddleX + Serving 插件）

    ```
    # Paddle（示例选择 cu118 通道；也可用官方 cu126 通道）
    uv pip install paddlepaddle-gpu==3.2.1 \
      -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

    # PaddleOCR（含 doc-parser 产线依赖）与 PaddleX
    uv pip install "paddleocr[doc-parser]" paddlex \
      -i https://pypi.tuna.tsinghua.edu.cn/simple

    paddlex --install serving
    ```

3.  安装 特制版 safetensors（Paddle 版）

    > 避免 framework paddle is invalid 的典型报错，按官方指引安装特别构建的 wheel（Linux 示例）：

        uv pip uninstall -y safetensors
        uv pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

    （PaddleOCR-VL RTX 50 环境教程与博文均强调需安装此特制版）

4.  生成并修改**产线配置**

        paddlex --get_pipeline_config PaddleOCR-VL > config/pipeline_config.yaml

    将 config/pipeline_config.yaml 的 VLRecognition.genai_config 指向 vLLM：

    ```
    VLRecognition:
      ...
      genai_config:
        backend: vllm-server
        server_url: http://127.0.0.1:8118/v1
        max_concurrency: 1

    ```

5.  统一模型缓存到本项目目录 (可选)

    ```
    export PADDLEX_HOME="$PWD/model/.paddlex"
    export PADDLEX_CACHE_DIR="$PADDLEX_HOME/temp"
    export PADDLE_HUB_HOME="$PADDLEX_HOME/official_models"
    export PPNLP_HOME="$PWD/model/.paddlenlp"
    export HF_HOME="$PWD/model/.cache/huggingface"
    export HUGGINGFACE_HUB_CACHE="$HF_HOME"
    ```

6.  启动 PaddleX 产线服务：8080 端口

        ```
        paddlex --serve --pipeline config/pipeline_config.yaml \
        --host 0.0.0.0 --port 8080 --device gpu
        ```

    **健康检查：**

    ```
    # 方式一：访问文档（若产线内置了 /docs）
    curl -s http://127.0.0.1:8080/docs >/dev/null && echo ok || echo fail

    # 方式二：用本仓库的自检脚本
    python runcheck.py data/paf/one-page.pdf
    ```

## 使用方法

> 图片或 PDF → POST /layout-parsing（已由 src/pipeline.py/runcheck.py 封装）

> 对长 PDF，runcheck.py 会自动按页/分块调用并逐页输出 Markdown（图片统一落在 <stem>\_assets/）：

1. LONG_PDF_PAGE_THRESHOLD（默认 40 页）
2. PAGES_PER_PART（默认 8 页/分块）
3. PADDLE_PDF_CHUNK_PAGES（默认 16，客户端提示服务端更省显存）

> 产线侧也可在请求体传入轻量化参数（如 maxPixels、vl_rec_max_concurrency 等）以减少显存峰值，示例见 runcheck.py。

## 参数与调优

> vLLM（后端）

1. --max-num-batched-tokens：吞吐与延迟折中

2. --no-enable-prefix-caching、--mm-processor-cache-gb 0：OCR/VL 场景推荐关闭无效缓存

3. 详见 vLLM OpenAI 兼容服务文档。

> PaddleX（产线）

1. VLRecognition.genai_config.max_concurrency：单机单服务建议从 1 起步，避免过载

2. 可在请求中关闭 visualize 以减少返回体 & 服务端渲染开销（亦可在配置里全局关闭）

## 常见问题（FAQ）

> safetensors_rust.SafetensorError: framework paddle is invalid

1. 说明你装的是通用版 safetensors。请卸载并安装 Paddle 专用 wheel（见上文步骤 B-3）。官方明确要求安装特制版本。

> 如何把模型/权重缓存到项目目录？

1. 为 Hugging Face 设置 HF_HOME/HUGGINGFACE_HUB_CACHE；为 ModelScope 设置 MODELSCOPE_CACHE；vLLM 侧再加 --download-dir ./model/hf 即可。

> 加速国内下载

1. 使用 清华 TUNA PyPI 镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple，或在 ~/.pip/pip.conf 中设置为默认。
