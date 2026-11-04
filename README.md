# 云桌面主 API + 远程三个 GPU 服务

复用你已经部署的 `POST /v1/audio/transcriptions`：通过 multipart 上传音频，传入 `model`、`response_format=json`、`stream=false`，读取响应的 `text`。本项目不再加载 Qwen3-ASR 权重。

```text
云桌面：Python 主 API :8010 / 持久化任务队列
         → 服务器 10.130.1.6:8001：已有 Qwen3-ASR，分块识别正文
         → 服务器 10.130.1.6:8002：Community-1，整段录音说话人聚类
         → 服务器 10.130.1.6:8003：ForcedAligner，分块字词时间戳
         → 主 API 恢复全局时间戳、匹配说话人、合并段落
```

代码直接平铺在当前文件夹。`app.py` 为对外 API，`pipeline.py` 为 HTTP 调度，`clients.py` 为三个服务的客户端，`gpu_services.py` 为两个可独立启动的 GPU 服务，`jobs.py` 为单节点 SQLite 持久化任务队列。

## 配置

配置分为两份：云桌面主 API 使用 `.env`（模板 `.env.example`），GPU 服务器上的 Compose 使用 `.env.gpu`（模板 `.env.gpu.example`）。服务器不需要主 API 的 `.env`，云桌面不需要模型权重和 GPU 配置。

云桌面的 `.env` 中填写三个服务的完整地址，以下沿用已验证的服务器 IP；若 GPU 服务分布在不同服务器，分别修改对应地址：

```dotenv
ASR_API_URL=http://10.130.1.6:8001/v1/audio/transcriptions
ASR_MODEL=
ASR_API_KEY=
DIARIZATION_API_URL=http://10.130.1.6:8002/v1/diarization
ALIGNER_API_URL=http://10.130.1.6:8003/v1/alignment
```

`ASR_API_URL` 必须包含完整接口路径。`ASR_MODEL` 留空时，首次转写会通过同一服务的 `GET /v1/models` 读取唯一模型 ID，之后在当前 API 进程内复用；此操作不会下载模型。若服务部署了多个模型或不支持模型列表接口，请明确填写与 Postman 成功请求完全一致的 `model`，可能是 `/data/...` 本地模型路径，而不是 Hugging Face 仓库名。已有服务不需要 API Key 时留空。云桌面通过服务器 IP 访问三个端口，不能使用 Docker 服务名 `diarization` / `aligner` 或云桌面的 `localhost`。

服务器的 `.env.gpu` 中配置本地模型和两张卡：

```dotenv
DIARIZATION_MODEL=/models/speaker-diarization-community-1
ALIGNER_MODEL=/models/Qwen3-ForcedAligner-0.6B
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIARIZATION_GPU_ID=1
ALIGNER_GPU_ID=2
DIARIZATION_DEVICE=cuda:0
ALIGNER_DEVICE=cuda:0
```

示例假设 ASR 已使用物理 GPU 0，另外两个服务使用物理 GPU 1、2。请按实际硬件修改 `*_GPU_ID`；每个容器只映射一张卡，因此容器内均为 `cuda:0`。可配置同一物理卡，但要为两个模型、推理临时张量及其他服务预留显存，并压测资源争用。

完整模型已下载后，无需配置 `HF_TOKEN` 或登录 Hugging Face；也无需配置 `HF_HOME` 缓存目录。保留 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 可直观看到离线设置。GPU 服务自身也固定启用离线模式，只接受包含模型配置文件的本地目录；路径或文件缺失时会报错，不会尝试在线补下载。`huggingface_hub`、Transformers 等 Python 库仍是模型加载所需依赖，应保留。

`compose.yaml` 只包含 `diarization`、`aligner` 两个 GPU 服务，共用 `meeting-gpu:offline` 镜像，分别映射服务器端口 `8002:8002`、`8003:8003`，保留模型挂载、GPU 选择和健康检查。已有 ASR 独立运行，不由这个 Compose 管理。调参项未填写时使用 `config.py` 默认值。

云桌面 `.env` 的 `SERVICE_API_KEY` 控制对外任务与转写接口；设置后调用时加 `Authorization: Bearer <key>`。如果启用 GPU 服务鉴权，云桌面 `.env` 与服务器 `.env.gpu` 中的 `INTERNAL_API_KEY` 必须相同；留空表示不开启。

主 API 直接在云桌面监听 8010，不需要构建 API 镜像。若服务器上仍运行旧版主 API 容器，请先用旧版 Compose 停止 `api`，或单独停止对应的旧主 API 容器，再使用新版配置启动 GPU 服务。

## 下载与离线部署

在外网接受 [Community-1 使用条件](https://huggingface.co/pyannote/speaker-diarization-community-1)，登录后完整下载两个模型：

```bash
python -m pip install 'huggingface_hub[cli]>=0.34,<1'
hf auth login
hf download pyannote/speaker-diarization-community-1 --local-dir ./models/speaker-diarization-community-1
hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./models/Qwen3-ForcedAligner-0.6B
```

保留 Community-1 的全部目录，包括 `config.yaml`、`embedding/`、`segmentation/` 和 `plda/`。本地完整权重加载见[官方说明](https://huggingface.co/pyannote/speaker-diarization-community-1#offline-use)。ASR 的权重由你现有的服务管理。

在与 GPU 服务器架构相同的联网 Docker 环境构建一次 GPU 镜像，供两个 GPU 容器共用：

```bash
docker build -f Dockerfile.gpu -t meeting-gpu:offline .
docker save -o meeting-gpu.tar meeting-gpu:offline
```

使用 `Dockerfile.gpu`，无需 `--target` 或 `docker compose build`。GPU 依赖单独放在 `requirements-gpu.txt`，由 GPU 镜像构建时安装；主项目不再声明 `gpu` 可选依赖。`Dockerfile.api` 保留为可选部署方式，当前方案不使用。若此前已打包旧镜像，本地目录校验、离线文档及固定离线加载逻辑需要重新构建、导入 GPU 镜像后生效。

将 `meeting-gpu.tar`、`models/`、`compose.yaml` 和配置好的 `.env.gpu` 拷到内网 GPU 服务器同一目录。服务器预先安装 Docker Compose 和 NVIDIA GPU 容器支持，驱动需兼容 CUDA 12.8 运行环境，然后执行：

```bash
docker load -i meeting-gpu.tar
docker compose --env-file .env.gpu up -d --no-build --pull never
docker compose --env-file .env.gpu ps
docker compose --env-file .env.gpu logs -f
```

必须保留 `--env-file .env.gpu`：它用于替换 Compose 中的物理 GPU 编号；服务的 `env_file: .env.gpu` 则把模型路径等设置传入容器。修改 GPU 配置后重新执行上面的启动命令会更新容器。

GPU 服务启动时加载模型；健康后，按后面的步骤启动云桌面主 API。GPU 文档分别位于 `http://10.130.1.6:8002/docs`、`http://10.130.1.6:8003/docs`，主 API 文档在云桌面的 `http://localhost:8010/docs`。本项目三个服务的 `/docs` 均使用随代码、镜像打包的 Swagger UI 静态文件，浏览器无需访问外网 CDN。API 的 `/health` 表示进程存活，不代表已验证远程服务可访问。

如果日志显示 `/docs` 为 200 但旧版页面空白，可更新主 API 代码或 GPU 镜像，再强制刷新浏览器。如果任务报 ASR HTTP 404，请先核对 `ASR_API_URL`，再核对 `ASR_MODEL` 是否与 `GET http://ASR服务器:8001/v1/models` 返回的 `data[].id` 完全一致。新版任务错误会附上服务名、HTTP 状态码和上游错误原因。修改云桌面的 `.env` 后重启 Python 主 API 进程。

## 对外调用

大批量调用使用异步任务接口。上传完整音频后立即得到任务 ID：

```bash
curl http://localhost:8010/v1/jobs \
  -F 'file=@meeting.wav' -F 'num_speakers=3'
```

人数未知时省略 `num_speakers`，也可设置 `min_speakers` / `max_speakers`。返回 HTTP 202：

```json
{"id":"任务ID","status_url":"/v1/jobs/任务ID","result_url":"/v1/jobs/任务ID/result"}
```

使用 **GET** 查询进度、获取你需要的纯文本；查询地址使用 POST 会返回 405：

```bash
curl http://localhost:8010/v1/jobs/任务ID
curl 'http://localhost:8010/v1/jobs/任务ID/result?response_format=text'
```

任务状态为 `uploading`、`queued`、`running`、`completed`、`failed`。结果尚未完成时返回 202，失败时结果接口返回 409，状态中的 `error` 提供原因。任务队列已满时提交返回 429 和 `Retry-After`。

同步调试接口仍然可用：

```bash
curl http://localhost:8010/v1/audio/transcriptions \
  -F 'file=@meeting.wav' -F 'num_speakers=3' -F 'response_format=text'
```

同步调用占用同一组处理槽位，满时返回 503。外部推理服务调用失败返回 502。Windows PowerShell 使用 `curl.exe`，并将上述命令写成一行或改用 PowerShell 换行符。

纯文本格式：

```text
发言人1
00:00
那咱这边

发言人2
00:01
这个东西有人看吗？
```

JSON 结果包含 `text`、`formatted_text`、`segments`、`words`、`diarization`、`speakers` 和 `warnings`。起止时间为相对原录音的秒数；中文通常每字一个时间戳。标点与空格保留在字词的 `text` 中。编号按首次出现排序，仅在当前录音内有效。重叠说话按 Community-1 互斥时间线分配一个主要说话人；本项目不做声源分离。

## 并发与扩容边界

默认最多两个任务执行、100 个未完成任务（包括上传、排队、执行），其余拒绝；这只是资源上限配置，不代表实测吞吐。两个 GPU 服务各自常驻一份模型，每进程一次推理，最多容纳四个处理中的请求；多任务能够交错使用三个推理服务。通过增加 GPU 服务实例并在其前方配置负载均衡扩容，调度端的 `DIARIZATION_API_URL` / `ALIGNER_API_URL` 指向负载均衡地址。

```dotenv
JOB_WORKERS=2
MAX_PENDING_JOBS=100
GPU_PENDING_REQUESTS=4
ASR_CHUNK_SECONDS=30
API_TIMEOUT_SECONDS=600
MAX_UPLOAD_MB=500
MAX_AUDIO_SECONDS=3600
```

`ASR_CHUNK_SECONDS` 默认 30 秒，在低能量位置切分且不丢失样本，不能超过 180 秒；说话人分离始终使用整段录音，保持统一编号。分块转写可能影响边界处的识别效果，应使用目标录音验证。增加执行任务数会增加 CPU 内存、解码和网络开销，GPU 服务的等待也计入请求超时；根据真实录音长度、各阶段延迟和显存压测调节。请求限流、TLS 和每用户配额应在入口网关配置。

SQLite 队列支持单节点持久化和重启恢复：运行中断的任务重新排队，未完成上传会清理。**每个 `JOB_DIRECTORY` 只能运行一个 API 进程，保持 `--workers 1`**。每个独立 GPU 服务也保持一个 worker，避免重复加载权重。当前队列不是多主机分布式调度系统；需要多个 API 副本共享全局队列时应替换为 Redis/RabbitMQ 等独立队列。

输入录音在任务完成或失败后删除。结果保存在 `outputs/jobs/jobs.sqlite3`，需由调用方在取走结果后执行 `DELETE /v1/jobs/任务ID` 清理；不会自动过期。部署时需为排队录音和结果预留磁盘空间。共享 `SERVICE_API_KEY` 不区分租户，用户间的任务授权隔离需要在业务层实现。

## 云桌面直接启动主 API

可以把项目源码直接复制到云桌面。保留 `.env`、Python 源文件、`swagger_assets/`、`pyproject.toml` 和 `uv.lock`；无需复制 `.venv/`、`.cache/`、`build/`、`dist/` 或 `models/`。虚拟环境应在目标云桌面重新创建，不能跨机器直接复用。

云桌面无需 GPU、模型权重或 Docker。安装 Python 3.10～3.13（建议 3.12）、uv 和 FFmpeg；配置 `.env` 后，在项目根目录执行：

```bash
uv sync
uv run --no-sync python -m uvicorn app:app --host 0.0.0.0 --port 8010 --workers 1
```

批处理也可直接通过 uv 运行：

```bash
uv run --no-sync python batch_transcribe.py "D:/录音" --output-dir "D:/转写结果"
```

`uv sync` 只安装主项目基础依赖，不解析或安装 `requirements-gpu.txt` 中的 PyTorch、CUDA、Qwen 和 pyannote 依赖。此前 GPU 依赖放在同一个项目的可选依赖中，uv 创建通用锁文件时会一起解析，即使默认同步时未选择安装它们。现在已经分离；不要再使用旧的 `uv sync --extra gpu` 或 `pip install '.[gpu]'`。

如果基础依赖已经通过离线包安装，也可从项目根目录直接运行：

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8010 --workers 1
```

在云桌面浏览器访问 `http://localhost:8010/docs`；其他客户端访问 `http://云桌面IP:8010`。确认云桌面能访问 GPU 服务器的 8001、8002、8003 端口。录音、任务队列和结果保存在云桌面，模型文件仅在 GPU 服务器上。

`uv sync` 首次安装需要访问可用的 Python 包源，单独复制源码不能替代依赖安装。完全离线的云桌面需提前准备依赖包；在与云桌面操作系统、Python 版本和架构匹配的联网环境准备 wheel 包：

```bash
python -m pip wheel --wheel-dir wheelhouse .
```

把 `wheelhouse/` 和项目拷到云桌面，执行：

```bash
python -m pip install --no-index --find-links=wheelhouse meeting-asr-server
```

FFmpeg 需另外准备并加入云桌面的 PATH。

如果报 `NumPy was built with baseline optimizations (X86_V2) but your machine doesn't support (X86_V2)`，说明云桌面当前 CPU 特性不满足所装 NumPy 的要求。NumPy 2.4 起提高了 x86 默认编译基线，部分虚拟机未暴露这些指令集。本项目将 NumPy 限制为 `>=2.0,<2.4`；更新 `pyproject.toml` 和 `uv.lock` 后执行 `uv sync`，再用 `uv run --no-sync python -c "import numpy; print(numpy.__version__)"` 确认导入正常，然后重跑批处理。此错误出现在云桌面的音频预处理阶段，不涉及远程模型推理。版本限制依据见 [NumPy 2.4 发布说明](https://numpy.org/doc/2.4/release/2.4.0-notes.html#modulate-dispatched-x86-cpu-features)。

## 文件夹批量处理

云桌面可直接运行 `batch_transcribe.py`，读取主 API 的 `.env` 并调用远程三个推理服务，无需先启动 8010 API：

```bash
python batch_transcribe.py "D:/录音" --output-dir "D:/转写结果"
```

默认处理当前文件夹的 WAV、MP3、M4A、FLAC、AAC、OGG、OPUS、WMA、AMR、AIFF 等音频，一次处理一个文件。包括子文件夹、同时处理两个文件、跳过已有结果：

```bash
python batch_transcribe.py "D:/录音" --output-dir "D:/转写结果" --recursive --workers 2 --skip-existing
```

每个音频保存两份结果，例如 `会议.wav.txt`（发言人、时间戳、正文）和 `会议.wav.json`（完整结果），保留输入子目录结构。文件名保留原扩展名，避免同名 WAV 和 MP3 相互覆盖。单文件失败后继续，结束时保存 `batch-summary.json`，列出每个文件的状态、输出位置和失败原因；有失败时脚本退出码为 1，全部成功或跳过时为 0。

`--skip-existing` 仅在对应 TXT、JSON 都存在时跳过；默认重新处理并覆盖对应输出。修改输入音频、模型或识别参数后，应省略此选项重新处理。还支持 `--language`、`--num-speakers`、`--min-speakers`、`--max-speakers`；文件间人数不同可不指定，让模型分别判断。每个音频的发言人编号独立。

`--workers` 控制本次脚本的并发文件数，独立于运行中的 API 进程；脚本和 API 同时运行时，远程服务承受两者合计的请求。此脚本是一次性批处理，不会持续监控目录，也不将任务写入 API 的 SQLite 队列。运行环境仍需基础 Python 依赖和 FFmpeg，模型和 GPU 继续位于服务器。

## 本地验证

联网开发环境可用 `pip install .` 安装主 API，GPU 服务的依赖由 `Dockerfile.gpu` 安装。主 API 使用 `.env`，GPU 容器通过 Compose 注入 `.env.gpu`，不会读取云桌面的配置文件。

直接调用 GPU 服务时，Community-1 使用 `POST http://10.130.1.6:8002/v1/diarization`，上传 `file`，可选说话人数；对齐使用 `POST http://10.130.1.6:8003/v1/alignment`，上传 `file`、`text` 和 `language=Chinese`。对齐服务每次音频上限 180 秒。可通过这两个接口分别进行模型效果与吞吐测试。

命令行输出文件：`python transcribe.py meeting.wav --num-speakers 3`，仍调用上述三个 HTTP 服务。

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m ruff check .
```

测试可在没有 GPU、没有模型权重的环境运行：验证 HTTP 文件上传协议、服务编排、队列恢复、错误处理、字词全局时间戳与 FFmpeg 解码。真实三服务联调、GPU 吞吐和 Docker 镜像构建需在你的部署环境中验证。
