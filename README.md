# 云桌面主 API + ASR / 说话人分段

复用你已经部署的 `POST /v1/audio/transcriptions`：通过 multipart 上传音频，传入 `model`、`response_format=json`、`stream=true`、`temperature=0` 和 `max_completion_tokens`，读取 SSE 转写事件并汇总为正文。流式读取用于提前停止异常生成及识别截断。本项目先用 Community-1 对整段录音进行说话人分段，再按其起止时间切音频、分别调用 ASR，不需要强制对齐模型。本项目不加载 Qwen3-ASR 权重。

```text
云桌面：Python 主 API :8010 / 持久化任务队列
         → 服务器 10.130.1.6:8002：Community-1，整段录音说话人分段与聚类
         → 主 API 按说话人时间段切音频，长片段再细分
         → 服务器 10.130.1.6:8001：已有 Qwen3-ASR，逐段识别正文
         → 主 API 使用说话人时间戳、合并段落
```

代码直接平铺在当前文件夹。`app.py` 为对外 API，`pipeline.py` 为 HTTP 调度，`clients.py` 为 ASR 和说话人服务的客户端，`transcript.py` 负责组装段落结果，`gpu_services.py` 为 Community-1 GPU 服务，`jobs.py` 为单节点 SQLite 持久化任务队列。

## 配置

配置分为两份：云桌面主 API 使用 `.env`（模板 `.env.example`），GPU 服务器上的 Compose 使用 `.env.gpu`（模板 `.env.gpu.example`）。服务器不需要主 API 的 `.env`，云桌面不需要模型权重和 GPU 配置。

云桌面的 `.env` 中填写 ASR 和说话人两个服务的完整地址，以下沿用已验证的服务器 IP；若 GPU 服务分布在不同服务器，分别修改对应地址：

```dotenv
ASR_API_URL=http://10.130.1.6:8001/v1/audio/transcriptions
ASR_MODEL=
ASR_API_KEY=
DIARIZATION_API_URL=http://10.130.1.6:8002/v1/diarization
```

`ASR_API_URL` 必须包含完整接口路径。`ASR_MODEL` 留空时，首次转写会通过同一服务的 `GET /v1/models` 读取唯一模型 ID，之后在当前 API 进程内复用；此操作不会下载模型。若服务部署了多个模型或不支持模型列表接口，请明确填写与 Postman 成功请求完全一致的 `model`，可能是 `/data/...` 本地模型路径，而不是 Hugging Face 仓库名。已有服务不需要 API Key 时留空。云桌面通过服务器 IP 访问 8001、8002 两个端口，不能使用 Docker 服务名 `diarization` 或云桌面的 `localhost`。

服务器的 `.env.gpu` 中配置 Community-1 本地模型和 GPU：

```dotenv
DIARIZATION_MODEL=/models/speaker-diarization-community-1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DIARIZATION_GPU_ID=1
DIARIZATION_DEVICE=cuda:0
```

示例假设 ASR 已使用物理 GPU 0，Community-1 使用物理 GPU 1。请按实际硬件修改 `DIARIZATION_GPU_ID`；容器只映射一张卡，因此容器内为 `cuda:0`。也可与 ASR 共用物理卡，但需根据实际显存和资源争用压测。

完整模型已下载后，无需配置 `HF_TOKEN` 或登录 Hugging Face；也无需配置 `HF_HOME` 缓存目录。保留 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 可直观看到离线设置。GPU 服务自身也固定启用离线模式，只接受包含模型配置文件的本地目录；路径或文件缺失时会报错，不会尝试在线补下载。GPU Python 依赖按 `requirements-gpu.txt` 安装即可。

`compose.yaml` 只包含 `diarization` 一个 GPU 服务，使用 `meeting-gpu:offline` 镜像，映射服务器端口 `8002:8002`，保留模型挂载、GPU 选择和健康检查。已有 ASR 独立运行，不由这个 Compose 管理。调参项未填写时使用 `config.py` 默认值。

云桌面 `.env` 的 `SERVICE_API_KEY` 控制对外任务与转写接口；设置后调用时加 `Authorization: Bearer <key>`。如果启用 GPU 服务鉴权，云桌面 `.env` 与服务器 `.env.gpu` 中的 `INTERNAL_API_KEY` 必须相同；留空表示不开启。

主 API 直接在云桌面监听 8010，不需要构建 API 镜像。若服务器上仍运行旧版主 API 容器，请先用旧版 Compose 停止 `api`，或单独停止对应的旧主 API 容器，再使用新版配置启动 GPU 服务。

## 下载与离线部署

在外网接受 [Community-1 使用条件](https://huggingface.co/pyannote/speaker-diarization-community-1)，登录后完整下载 Community-1 模型：

```bash
python -m pip install 'huggingface_hub[cli]>=0.34,<1'
hf auth login
hf download pyannote/speaker-diarization-community-1 --local-dir ./models/speaker-diarization-community-1
```

保留 Community-1 的全部目录，包括 `config.yaml`、`embedding/`、`segmentation/` 和 `plda/`。本地完整权重加载见[官方说明](https://huggingface.co/pyannote/speaker-diarization-community-1#offline-use)。ASR 的权重由你现有的服务管理。

在与 GPU 服务器架构相同的联网 Docker 环境构建 Community-1 GPU 镜像：

```bash
docker buildx version
docker buildx build --load --progress=plain -f Dockerfile.gpu -t meeting-gpu:offline .
docker save -o meeting-gpu.tar meeting-gpu:offline
```

如果提示 `legacy builder is deprecated` 且 `docker buildx` 不存在，请先安装与你的 Docker 来源匹配的 Buildx。Ubuntu 通过发行版仓库安装 `docker.io` 时，对应包是 `docker-buildx`：`sudo apt-get install --no-install-recommends docker-buildx`；Docker 官方 CE 仓库使用的包名则是 `docker-buildx-plugin`。安装后运行 `docker buildx version` 确认，再执行上面的构建命令，无需为安装 CLI 插件重启 Docker。

WSL 中建议从 Linux 文件系统构建。若项目在 `/mnt/c` 或 `/mnt/d`，可以在项目目录执行下面命令，只复制源码和构建所需文件，再从副本构建；模型、录音、配置密钥和虚拟环境无需复制：

```bash
task_build_dir=$(mktemp -d /tmp/asr-gpu-build.XXXXXXXX)
cp -a Dockerfile.gpu .dockerignore pyproject.toml requirements-gpu.txt *.py swagger_assets "$task_build_dir/"
cd "$task_build_dir"
docker buildx build --load --progress=plain -f Dockerfile.gpu -t meeting-gpu:offline .
```

`--load` 将构建结果加载到本机 Docker，供后续 `docker save` 和 Compose 使用；`--progress=plain` 输出完整阶段和耗时。旧构建器停在 `COPY` 时，应先检查 Docker 服务响应和存储状况；该步骤尚未执行 Python 依赖安装。不要仅凭 `df` 中 WSL 虚拟磁盘的最大余量判断宿主机物理盘空间充足。

使用 `Dockerfile.gpu`，无需 `--target` 或 `docker compose build`。GPU 依赖单独放在 `requirements-gpu.txt`，由 GPU 镜像构建时安装；主项目不再声明 `gpu` 可选依赖。`Dockerfile.api` 保留为可选部署方式，当前方案不使用。若此前已打包旧镜像，本地目录校验、离线文档及固定离线加载逻辑需要重新构建、导入 GPU 镜像后生效。

将 `meeting-gpu.tar`、`models/`、`compose.yaml` 和配置好的 `.env.gpu` 拷到内网 GPU 服务器同一目录。服务器预先安装 Docker Compose 和 NVIDIA GPU 容器支持，驱动需兼容 CUDA 12.8 运行环境，然后执行：

```bash
docker load -i meeting-gpu.tar
docker compose --env-file .env.gpu up -d --no-build --pull never
docker compose --env-file .env.gpu ps
docker compose --env-file .env.gpu logs -f
```

必须保留 `--env-file .env.gpu`：它用于替换 Compose 中的物理 GPU 编号；服务的 `env_file: .env.gpu` 则把模型路径等设置传入容器。修改 GPU 配置后重新执行上面的启动命令会更新容器。

GPU 服务启动时加载模型；健康后，按后面的步骤启动云桌面主 API。GPU 文档位于 `http://10.130.1.6:8002/docs`，主 API 文档在云桌面的 `http://localhost:8010/docs`。本项目两个服务的 `/docs` 均使用随代码、镜像打包的 Swagger UI 静态文件，浏览器无需访问外网 CDN。API 的 `/health` 表示进程存活，不代表已验证远程服务可访问。

如果日志显示 `/docs` 为 200 但旧版页面空白，可更新主 API 代码或 GPU 镜像，再强制刷新浏览器。如果任务报 ASR HTTP 404，请先核对 `ASR_API_URL`，再核对 `ASR_MODEL` 是否与 `GET http://ASR服务器:8001/v1/models` 返回的 `data[].id` 完全一致。新版任务错误会附上服务名、HTTP 状态码和上游错误原因。修改云桌面的 `.env` 后重启 Python 主 API 进程。

## 从带对齐模型的版本升级

先同步新版云桌面源码和 `.env`，重启主 API 或批处理。输出仍支持发言人、段落时间和正文；完整 JSON 的 `words` 为空，`timestamp_mode` 为 `segment`。已有 Community-1 和 ASR 服务可以继续使用，无需为了云桌面的流程修改立即重建镜像。

在 GPU 服务器替换旧 `compose.yaml` 之前，用旧配置停止并移除对齐容器：

```bash
docker compose --env-file .env.gpu stop aligner
docker compose --env-file .env.gpu rm -f aligner
```

然后同步新版 `compose.yaml` 和 `.env.gpu`，按前文启动 `diarization`。若已替换旧 Compose 文件，可通过 `docker ps -a` 确认旧对齐容器的名称，单独停止并移除该容器。

新版流程验证成功、旧对齐容器停止后，可以删除服务器上的 `./models/Qwen3-ForcedAligner-0.6B` 模型目录，释放磁盘空间。保留 `speaker-diarization-community-1` 和现有 Qwen3-ASR 权重。新版源码和 GPU 依赖不再需要 `qwen-asr` 包；若要精简旧 GPU 镜像，需重建、导入镜像后重新创建 `diarization` 容器。现有独立 ASR 服务仍需要自己的 Qwen 环境。

用 `--resume-dir "旧结果目录"` 会继续复用有效旧结果；需要验证新流程或重新生成段落时间戳时，正常启动新批次即可。

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

JSON 结果包含 `text`、`formatted_text`、`segments`、`diarization`、`speakers` 和 `warnings`。新增 `timestamp_mode: "segment"` 表示段落级时间戳；兼容字段 `words` 固定为空数组，不生成字词级时间戳。起止时间沿用 Community-1 说话人片段，单位为相对原录音的秒数；相邻同一说话人片段可按 `MERGE_GAP_SECONDS` 合并，保留首段开始和末段结束。标点与空格保留在正文中。编号按首次出现排序，仅在当前录音内有效。重叠说话使用 Community-1 互斥时间线的主要说话人，音频本身仍是混音，本项目不做声源分离。

## 并发与扩容边界

默认最多两个任务执行、100 个未完成任务（包括上传、排队、执行），其余拒绝；这只是资源上限配置，不代表实测吞吐。Community-1 服务常驻一份模型，每进程一次推理，最多容纳四个处理中的请求；多任务能够交错使用 ASR 和说话人两个推理服务。通过增加服务实例并在其前方配置负载均衡扩容，调度端的对应 API URL 指向负载均衡地址。

```dotenv
JOB_WORKERS=2
MAX_PENDING_JOBS=100
GPU_PENDING_REQUESTS=4
ASR_CHUNK_SECONDS=30
API_TIMEOUT_SECONDS=600
MAX_UPLOAD_MB=500
MAX_AUDIO_SECONDS=3600
```

`ASR_CHUNK_SECONDS` 默认 30 秒，限制每个说话人片段内的 ASR 子块长度；长片段在低能量位置切分且不丢失片段内样本，识别后合并回原说话人片段。不同说话人分别调用 ASR，不跨说话人拼接音频。已取消对齐模型带来的 180 秒配置上限，实际值应结合 ASR 服务限制和生成预算设置。主流程先解码为 16 kHz 单声道，另存一份整段 PCM16 临时 WAV 上传给说话人服务，聚类完成后删除临时文件，保持整段录音的统一编号与原始时间轴；不再上传原始多通道大文件。16 kHz、16 位、8 通道音频总码率为 2048 kbps，每小时约 879 MiB；同采样率的单声道 PCM16 每小时约 110 MiB，默认一小时录音可在 GPU 服务的 500 MiB 文件限制内上传。对外 API 的原始上传大小仍由 `MAX_UPLOAD_MB` 控制。

在调用 ASR 前检查说话人时间段：必须有限、有效、位于录音范围内且互斥；仅容忍末尾不超过 0.5 秒的越界并裁到录音结尾，其他异常直接报错。每个 ASR 分块的正文上限为每秒 40 字符，至少允许 256 字符，最多 4096 字符，用于拦截异常长文本。文本异常过长或 ASR 明确返回截断时，只将当前异常音频块二等分重新识别；最多拆分 3 层，仅在当前块不少于 10 秒时拆分，子块不小于 5 秒。先前块不重复处理，异常块的部分文本不会混入成功输出，恢复情况记录在 `.full.json` 的 `warnings` 中。达到重试边界后仍失败时，错误包含分块的原录音位置；说话人片段时间戳不因 ASR 细分或重试改变。网络错误不触发音频细分。

说话人边界切音频及异常块重新识别可能影响切点处的识别效果，短应答缺少上下文，重叠讲话仍可能混入其他人的文字，应使用目标录音验证。此方案省去对齐推理，但碎片较多时 ASR 请求数会增加，实际总耗时需实测。增加执行任务数会增加 CPU 内存、解码和网络开销，GPU 服务的等待也计入请求超时；根据真实录音长度、各阶段延迟和显存压测调节。请求限流、TLS 和每用户配额应在入口网关配置。

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

`uv sync` 只安装主项目基础依赖，不解析或安装 `requirements-gpu.txt` 中的 PyTorch、CUDA 和 pyannote 依赖。此前 GPU 依赖放在同一个项目的可选依赖中，uv 创建通用锁文件时会一起解析，即使默认同步时未选择安装它们。现在已经分离；不要再使用旧的 `uv sync --extra gpu` 或 `pip install '.[gpu]'`。

如果基础依赖已经通过离线包安装，也可从项目根目录直接运行：

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8010 --workers 1
```

在云桌面浏览器访问 `http://localhost:8010/docs`；其他客户端访问 `http://云桌面IP:8010`。确认云桌面能访问 GPU 服务器的 8001、8002 端口。录音、任务队列和结果保存在云桌面，模型文件仅在 GPU 服务器上。

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

云桌面可直接运行 `batch_transcribe.py`，读取主 API 的 `.env` 并调用远程 ASR 和说话人两个推理服务，无需先启动 8010 API：

```bash
python batch_transcribe.py "D:/录音" --output-dir "D:/转写结果"
```

`--output-dir` 是输出父目录；每次正常启动都会新建“北京时间戳 + 8 位唯一标识”的批次子目录，在启动和结束时打印完整位置。不同批次不会覆盖结果，默认日志也各自保存。例如：

```text
D:/转写结果/
└── 20261105_203015_a1b2c3d4/
    ├── batch.log
    ├── batch-summary.json
    ├── timing-report.txt
    ├── 会议.wav.txt
    ├── 会议.wav.json
    └── 会议.wav.full.json
```

默认处理当前文件夹的 WAV、MP3、M4A、FLAC、AAC、OGG、OPUS、WMA、AMR、AIFF 等音频，一次处理一个文件。包括子文件夹、同时处理两个文件：

```bash
python batch_transcribe.py "D:/录音" --output-dir "D:/转写结果" --recursive --workers 2
```

每个音频保存以下结果，并保留输入子目录结构：

- `会议.wav.txt`：保持原来的发言人、时间戳、正文排版，方便阅读。
- `会议.wav.json`：与 TextGrid 转换结果相同的片段数组，每项仅包含 `speaker`、`start`、`end`、`text`，时间单位为秒，按开始时间排序，方便对比。
- `会议.wav.full.json`：完整识别结果，包含 `segments`、段落时间戳、说话人时间线、耗时和警告（`words` 为空），方便排查问题。

文件名保留原音频扩展名，避免同名 WAV 和 MP3 相互覆盖。单文件失败后继续，结束时在实际批次目录保存 `batch-summary.json` 和 `timing-report.txt`，列出每个文件的状态、输出位置、开始与结束时间、耗时和失败原因；有失败时脚本退出码为 1，全部成功或跳过时为 0。同步 API 和单文件 `transcribe.py` 仍输出完整结果对象。

续跑时使用 `--resume-dir` 指向已有的实际批次目录，自动启用 `--skip-existing`，复用对应 TXT 和有效 JSON，不新建批次目录：

```bash
python batch_transcribe.py "D:/录音" --resume-dir "D:/转写结果/20261105_203015_a1b2c3d4" --recursive --workers 2
```

旧版没有时间戳子目录的结果，也可以直接用 `--resume-dir "D:/旧转写结果"` 续跑。如果 JSON 来自旧版本的完整结果对象，会将其转为片段数组，并把原完整内容保存到 `.full.json`，不重复调用推理服务，TXT 保持原样；无效 JSON 或缺少 TXT/JSON 时重新处理。续跑会追加日志，汇总和耗时报告更新为本次执行的数据；跳过文件不会沿用旧模型耗时参与本次平均。单独给新批次加 `--skip-existing` 无法复用其他批次的结果。修改输入音频、模型或识别参数后，正常启动新批次重新处理。

耗时报告包括批次实际总耗时、每个文件的耗时、实际处理文件平均耗时（含失败、不含跳过）、成功文件平均耗时，以及解码、说话人分离、ASR、结果组装和写入各阶段的累计与平均耗时。单文件总耗时从该文件开始执行计到结果写入或失败，不包含线程池排队时间；批次总耗时包含文件扫描、排队与处理，统计到所有文件结束，不包含最终报告的落盘。并发时各文件耗时之和可能大于批次总耗时。阶段平均值仅使用有记录的数据，逐项显示统计文件数；上游失败会保留已完成阶段和失败阶段截至报错的耗时，未执行阶段显示“未记录”，不按零秒参与平均。阶段耗时包含云桌面的传输、服务等待、推理及重试，不表示纯 GPU 计算耗时。

`batch-summary.json` 的 `timing_summary` 保存实际处理、成功文件和各阶段的 `count`、`total_seconds`、`average_seconds`，无有效样本时平均值为 `null`。逐文件记录还包含 `duration_seconds`、`processing_seconds` 和 `timings`；`.full.json` 也包含流水线的分阶段 `timings`。还支持 `--language`、`--num-speakers`、`--min-speakers`、`--max-speakers`；文件间人数不同可不指定，让模型分别判断。每个音频的发言人编号独立。

终端会显示当前文件、音频时长、分块时间范围、说话人分段和逐段 ASR 阶段，以及请求开始、耗时和重试次数。如果 15 秒内没有文件完成，会打印当前处理的文件与本批次耗时。HTTP 200 只代表该次请求成功，不能据此判断整段录音已经处理完；同一录音还可能继续请求后续分块。

批处理默认把同样的日志追加写入实际批次目录下的 `batch.log`，包含线程名、Python 路径、配置摘要和失败位置。也可用 `--log-file` 显式指定其他日志路径：

```bash
uv run --no-sync python batch_transcribe.py "D:/录音" --output-dir "outputs/transcribe" --log-file "outputs/transcribe/debug.log"
```

排查“服务器已返回 200，但客户端没有进入 ASR”时，依次查看以下阶段；网络传输日志由 HTTP 客户端产生，`收到响应头` 与 `正文已收完` 分开记录：

```text
Community-1 请求体发送完成，等待服务响应
Community-1 收到响应头，HTTP 200；即将读取正文，尚不代表接收完成
Community-1 响应正文已收完；开始解析 JSON
Community-1 JSON 解析完成
Community-1 结果读取完成，共 ... 个说话人时间段
说话人响应已转换；开始清理临时 WAV
临时 WAV 清理完成；开始校验说话人时间段
说话人片段 ... 开始 ASR
Qwen3-ASR 收到响应头，HTTP 200；即将读取流式正文
Qwen3-ASR 收到首条 SSE 数据
Qwen3-ASR 流式正文读取完成
```

ASR 在收到流式事件时每隔约 15 秒记录一次事件数、字符数和耗时；无新事件时不会伪造接收进度，批处理仍会打印等待信息。阶段日志不打印识别正文、鉴权头或网络跟踪元数据。

当前 Community-1 接口先完成整段模型推理，再返回 JSON，因此对应的 `POST /v1/diarization` 出现 200 通常表示推理已经返回，但客户端是否读完正文、清理完临时文件仍要看后续日志。新版 GPU 服务还会分别记录解码耗时、等待推理锁的时间和实际模型推理耗时；需要同步 `gpu_services.py` 并重建、导入 GPU 镜像后才能看到这些服务端日志。仅增加云桌面日志时，更新 `clients.py`、`pipeline.py`、`batch_transcribe.py` 并重启批处理即可，不要求重建 GPU 镜像。

如果 Windows 传统控制台连每 15 秒的等待日志也停止，且按键后立刻恢复，先按 `Esc` 退出文本选择，再到窗口标题栏的“属性 → 选项”关闭“快速编辑模式”。控制台选择可能阻塞输出，不能仅凭截图认定是网络或 GPU 问题；参见 [Microsoft 控制台问题说明](https://github.com/microsoft/terminal/issues/2678)。不要把反复按 `Ctrl+C` 当作继续执行，因为它也可能触发程序中断。

Community-1 的读写等待超时由 `.env` 的 `API_TIMEOUT_SECONDS` 控制，默认 600 秒，连接异常或部分临时 HTTP 错误最多尝试 3 次。可以在本次批处理中覆盖公共超时，例如：

```bash
python batch_transcribe.py "D:/录音" --resume-dir "D:/转写结果/20261105_203015_a1b2c3d4" --timeout-seconds 120
```

ASR 单独使用 `ASR_TIMEOUT_SECONDS`，默认 90 秒，实际等待上限取它和公共超时的较小值；每收到一行流式数据也检查已耗时，持续生成超过该设置时关闭响应。HTTP 读写超时限制的是一次网络等待，并非严格的整段录音总时限。ASR 连接失败或超时不自动重复提交，避免旧生成尚未取消时继续堆积请求。若终端停留在某个阶段，结合对应服务日志检查该请求。

如果某个约 30 秒的块持续几分钟不结束，而 vLLM 日志中 `Running: 1 reqs`、生成 token 数和 KV cache 持续增长，说明模型仍在生成，可能出现重复输出或未正常生成结束标记。客户端默认每次请求最多生成 1024 tokens，并在流式正文超过该块字符预算时提前关闭响应。达到 token 上限（`finish_reason=length`）或正文异常长时，会将该音频块对半切分后重新识别，最多细分 3 层，最小子块约 5 秒；仍异常则报告文件失败，保留失败原因，不把截断正文当作完整结果。断流或没有正常结束标记同样报错。

可在云桌面 `.env` 中调整 ASR 请求限制：

```dotenv
ASR_MAX_COMPLETION_TOKENS=1024
ASR_TIMEOUT_SECONDS=90
```

现有 ASR 容器和模型可以继续使用。关键是每次转写请求明确携带 `max_completion_tokens`，不能只依赖启动时的默认生成配置：例如 vLLM 0.14.0 的转写实现，在请求未指定该参数时使用 `max_model_len` 作为生成预算。请求长度控制及流式结束原因见 [vLLM 0.14.0 转写源码](https://github.com/vllm-project/vllm/blob/v0.14.0/vllm/entrypoints/openai/speech_to_text.py)。本项目要求转写接口支持 SSE；若服务忽略 `stream=true`，会明确报错，避免静默接收无法判断是否截断的 JSON。

若旧请求仍在持续生成，先停止旧批处理，再在已有 `qwen3-asr` 容器中停止当前 `qwen-asr-serve` 进程，重新启动以清除旧生成；不要同时启动第二个进程占用同一 GPU/端口。可继续用原来的模型路径和显存配置，监听地址使用 vLLM 的 `--host` 参数：

```bash
CUDA_VISIBLE_DEVICES=0 qwen-asr-serve \
  /data/shared/Qwen3-ASR/models/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 --port 80
```

这里 1024 是本项目每个音频块的输出 token 上限，不表示音频秒数。按说话人时间段转写需要同步新版云桌面源码，尤其是 `pipeline.py`、`transcript.py`、`clients.py`、`config.py` 和 `pyproject.toml`，再停止旧批处理或重启主 API。已部署的 Community-1 接口仍兼容，可继续使用；新建精简镜像时再按前文重建 GPU 镜像。用 `--resume-dir` 指定旧批次目录可复用已完成文件；尚未完成的文件从头处理，不从中间分块续跑。已提交的旧请求不会自动获得新的生成上限，关闭客户端响应后还应查看 ASR 日志，确认旧生成已经取消。

如果遇到 AliMeeting 原始多通道 WAV 上传 Community-1 时返回 HTTP 413，请将更新后的 `pipeline.py` 和 `clients.py` 同步到云桌面实际运行目录后，结束旧批处理，再用 `--resume-dir` 指定旧结果目录重新启动。主流程会上传整段单声道 PCM16，不需要单纯调大 GPU 上传限制。处理失败的文件会重试，已有有效 TXT/JSON 的文件会复用。

`--workers` 控制本次脚本的并发文件数，独立于运行中的 API 进程；脚本和 API 同时运行时，远程服务承受两者合计的请求。此脚本是一次性批处理，不会持续监控目录，也不将任务写入 API 的 SQLite 队列。运行环境仍需基础 Python 依赖和 FFmpeg，模型和 GPU 继续位于服务器。

## AliMeeting 标注转 JSON

`textgrid_to_json.py` 仅使用 Python 标准库，无需安装额外依赖、配置远程服务或加载模型。转换单个标注文件：

```bash
python textgrid_to_json.py "D:/AliMeeting/会议.TextGrid" --output "outputs/reference/会议.json"
```

递归转换整个 Eval 数据目录，每个 TextGrid 生成一个 JSON，保留输入子目录结构：

```bash
python textgrid_to_json.py "D:/AliMeeting/Eval_Ali" --output-dir "outputs/reference/Eval_Ali"
```

也可只传入 `Eval_Ali_far/textgrid_dir`，只转换远场标注。默认输出目录为 `outputs/reference`；同名输出会覆盖。单个文件失败时继续处理其他文件，打印失败路径和原因，最终退出码为 1；全部成功为 0。

每个 JSON 是片段数组，与批处理 `会议.wav.json` 使用相同结构；同步 API 或 `.full.json` 则从 `segments` 提取对应片段。例如：

```json
[
  {"speaker": "SPK0093", "start": 1.25, "end": 4.8, "text": "我们先讨论一下今天的安排"},
  {"speaker": "SPK0094", "start": 4.2, "end": 6.1, "text": "好的我先说一下"}
]
```

起止时间沿用原录音时间轴，单位为秒。跳过空白文本区间和没有时长的点标注层（TextTier），按开始时间排序，保留不同说话人的重叠发言及文本中的标点、特殊标记。AliMeeting 层名中的 `SPKxxxx` 提取为说话人 ID；近场层名不含 ID 时尝试从文件名提取；其他层名原样作为说话人。支持 Praat 完整/短文本格式、双引号转义及跨行文本，自动读取 UTF-8（含 BOM）和带 BOM 的 UTF-16；旧编码可用 `--encoding gb18030` 等指定。二进制 TextGrid 需先在 Praat 中另存为文本。

人工标注的 `SPKxxxx` 与模型输出的聚类 ID、`发言人1` 等编号没有直接对应关系，对比说话人归属前需要按当前录音建立映射。参考标注按原始片段保存，模型结果可能合并多个片段，对比文本或时间戳时应按时间轴对齐。远场 WAV 文件名还可能包含额外的麦克风后缀，匹配 JSON 与音频时请核对会议编号。

## 识别结果与 reference 对比

`compare_transcripts.py` 只使用 Python 标准库，无需 GPU 或安装额外依赖。比较远场识别结果和之前生成的远场 reference，Windows PowerShell/CMD 均可运行下面这一行：

```bash
python compare_transcripts.py --reference "outputs/reference/Eval_Ali/Eval_Ali_far/textgrid_dir" --hypothesis "outputs/transcribe/20261105_203015_a1b2c3d4" --output-dir "outputs/comparison"
```

将 `--hypothesis` 改成某一个实际批次的输出目录，避免同时读入多个批次的同名结果；脚本递归搜索 JSON。也支持直接指定两个 JSON 文件，不要求文件名一致：

```bash
python compare_transcripts.py --reference "outputs/reference/Eval_Ali/Eval_Ali_far/textgrid_dir/R8001_M8004.json" --hypothesis "outputs/transcribe/20261105_203015_a1b2c3d4/R8001_M8004_MS801.wav.json" --output-dir "outputs/comparison"
```

比较逻辑不要求段数、边界或说话人名称一致：

- 自动匹配 `R8001_M8004_MS801.wav.json` 与 `R8001_M8004.json`，也支持普通同名录音；近场的 `_N_SPKxxxx` 不会当作远场参考。存在多个候选时报告歧义，不随意选取。
- 每份录音独立按时间重合总量建立最优的一一说话人映射，例如 `SPEAKER_00 → SPK8013`，不用编号本身判断对错。多识别或少识别的说话人不会被丢弃。
- **按时间映射的分说话人 CER**：将映射后的同一说话人文本按开始时间拼接，再计算字符编辑距离。标注分成三段、识别合成一段时，只要文字相同，CER 就为 0；不按行逐段配对。该指标同时反映文字和说话人归属，不是忽略说话人的全文 CER，也不是按文本另求最优排列的 cpCER。
- 默认归一化全半角、大小写，忽略标点、符号和空白；保留汉字、字母、数字、组合标记和“嗯、啊”等语气词。不自动转换简繁体、数字读法，也不自动删除标注中的噪声标签。确需排除方括号/尖括号标签时，可显式加 `--ignore-pattern "\[[^\]]*\]|<[^>]*>"`，同样应用于参考和识别正文。
- **片段时间线 DER**：计算漏检、误检、说话人混淆的时长之和，除以参考说话人总时长。采用零边界容差，包含重叠讲话；同一说话人自身重叠的区间只计一次。只改变分段边界而覆盖时间不变，不增加 DER；若合并片段填补了参考中的静音间隔，这段额外覆盖会计入误检。

这里 DER 评估的是输入 `segments` 的时间线。主流程会合并发言片段，因此它与 `.full.json` 内原始 `diarization` 的独立模型评分不同；即使传完整结果对象，也统一从 `segments` 取数据。本工具用于当前流程诊断，不代替 AliMeeting 官方评测协议。CER 定义参见 [TorchMetrics 文档](https://lightning.ai/docs/torchmetrics/stable/text/char_error_rate)，说话人映射及 DER 定义参见 [pyannote.metrics 文档](https://pyannote.github.io/pyannote-metrics/reference.html#pyannote.metrics.diarization.DiarizationErrorRate)。

输出目录包含：

- `comparison-report.txt`：汇总、每个文件的 CER/DER、说话人映射，以及每位说话人的完整参考/识别正文，方便人工对照。
- `comparison-report.json`：详细映射、各说话人编辑错误数、漏检/误检/混淆秒数及警告。
- `comparison-files.csv`：每个文件一行的指标表，UTF-8 BOM 编码，可用 Excel 打开；CER/DER 数值为比例，例如 `0.12` 表示 12%。

批处理的 `.wav.json` 和完整结果对象均可读；同一录音同时存在 `.wav.json` 和 `.wav.full.json` 时，优先使用前者，避免重复统计。缺少参考、缺少识别结果、重复匹配或无效 JSON 都会写入报告，不会静默跳过。选远场参考目录可避免把近场参考误报为缺少识别结果。

汇总 CER 按参考字符总量加权，DER 按参考说话人总时长加权，均只统计成功配对的文件，并单独列出缺失/失败数量；不是直接平均文件百分比。错误率可能超过 100%；参考文字或说话人时长为零时，对应比率为 `null`/`N/A`，仍保留错误数量和误检时长。有未配对或无效文件时退出码为 1，全部完成对比为 0；错误率大于零本身不导致命令失败。

## 本地验证

联网开发环境可用 `pip install .` 安装主 API，GPU 服务的依赖由 `Dockerfile.gpu` 安装。主 API 使用 `.env`，GPU 容器通过 Compose 注入 `.env.gpu`，不会读取云桌面的配置文件。

直接调用 GPU 服务时，Community-1 使用 `POST http://10.130.1.6:8002/v1/diarization`，上传 `file`，可选说话人数；响应 `turns` 中的每项包含 `start`、`end`、`speaker`，已经自带说话人起止时间，无需对齐模型补充。

命令行输出文件：`python transcribe.py meeting.wav --num-speakers 3`，调用 ASR 和 Community-1 两个 HTTP 服务。

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m ruff check .
```

测试可在没有 GPU、没有模型权重的环境运行：验证 HTTP 文件上传协议、服务编排、队列恢复、错误处理、按说话人切片、段落时间戳与 FFmpeg 解码。真实两服务联调、GPU 吞吐和 Docker 镜像构建需在你的部署环境中验证。
