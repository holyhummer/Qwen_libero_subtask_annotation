# StarVLA/native LIBERO 数据集批量标注脚本说明

本文档说明 `tools/annotate_starvla_libero_dataset_qwen35.py` 的用途、输入数据格式、运行方式和输出结构。文档中的路径均使用占位符表示，实际运行时请通过命令行参数传入本机路径。

## 脚本用途

`tools/annotate_starvla_libero_dataset_qwen35.py` 用于对 StarVLA/native-style LIBERO 数据集进行批量 sub-task 标注。它复用单 episode 脚本中的核心标注逻辑，包括：

- Qwen VLM 语义理解和 JSON 标注生成；
- 初始画面语言消歧；
- action/state 边界先验；
- action-aware 关键帧选择；
- 特殊 close 动作边界 refine；
- 校验、重试、错误记录和进度条显示。

该脚本面向完整数据集运行，每个 episode 会单独生成一个 annotation 目录。

## 输入数据结构

脚本期望 `--data-root` 下包含一个或多个以 `_lerobot` 结尾的 subset 目录，例如：

```text
<DATA_ROOT>/
  libero_10_no_noops_1.0.0_lerobot/
    data/
      chunk-000/
        episode_000000.parquet
        ...
    videos/
      chunk-000/
        observation.images.image/
          episode_000000.mp4
          ...
        observation.images.wrist_image/
          episode_000000.mp4
          ...
    meta/
      info.json
      tasks.jsonl
  libero_object_no_noops_1.0.0_lerobot/
  libero_goal_no_noops_1.0.0_lerobot/
  libero_spatial_no_noops_1.0.0_lerobot/
```

当前标注只使用 top-view 图像：

```text
observation.images.image
```

wrist/hand camera 不会送入 Qwen。

## 环境

建议为标注脚本单独创建 conda 环境，例如：

```bash
conda create -n libero_annot_qwen35 python=3.12 -y
conda activate libero_annot_qwen35
```

安装 PyTorch 时请根据本机 CUDA 版本选择对应命令。例如 CUDA 12.6 环境可使用：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

如果使用 CPU 或其他 CUDA 版本，请替换为对应的 PyTorch 安装命令。

安装标注脚本需要的其余包：

```bash
pip install \
  transformers \
  accelerate \
  safetensors \
  huggingface-hub \
  pandas \
  pyarrow \
  numpy \
  pillow \
  av \
  imageio \
  tqdm
```

当前脚本已验证可用的一组版本参考如下：

```text
torch 2.7.1+cu126
transformers 5.3.0
accelerate 1.13.0
safetensors 0.7.0
huggingface-hub 1.6.0
pandas 2.3.3
pyarrow 23.0.1
numpy 2.2.6
pillow 12.0.0
av 15.1.0
imageio 2.37.3
tqdm 4.67.3
```

安装完成后可检查脚本参数：

```bash
conda run -n libero_annot_qwen35 python tools/annotate_starvla_libero_dataset_qwen35.py --help
```

## 快速运行

对 `libero_10` 中的单个 episode 进行测试：

```bash
conda run -n libero_annot_qwen35 python tools/annotate_starvla_libero_dataset_qwen35.py \
  --data-root <DATA_ROOT> \
  --output-dir <OUTPUT_DIR> \
  --subsets libero_10_no_noops_1.0.0_lerobot \
  --episode-ids 15 \
  --overwrite
```

对某个 subset 的前 20 个 episode 运行：

```bash
conda run -n libero_annot_qwen35 python tools/annotate_starvla_libero_dataset_qwen35.py \
  --data-root <DATA_ROOT> \
  --output-dir <OUTPUT_DIR> \
  --subsets libero_10_no_noops_1.0.0_lerobot \
  --episode-start 0 \
  --episode-end 20 \
  --overwrite
```

对 `--data-root` 下所有 `_lerobot` subset 运行：

```bash
conda run -n libero_annot_qwen35 python tools/annotate_starvla_libero_dataset_qwen35.py \
  --data-root <DATA_ROOT> \
  --output-dir <OUTPUT_DIR>
```

如果只想试跑少量 episode，可以使用：

```bash
--limit 10
```

## 常用参数

```text
--data-root
    StarVLA/native LIBERO 数据根目录。建议显式传入，避免依赖脚本内默认路径。

--output-dir
    标注输出目录。建议显式传入。

--subsets
    逗号分隔的 subset 目录名。未指定时默认扫描 --data-root 下所有 *_lerobot 目录。

--episode-ids
    逗号分隔的 episode id 列表，例如 0,1,2,15。

--episode-start / --episode-end
    指定 episode 范围，end 为 exclusive。

--limit
    对本次选择到的 episode 数量进行截断，适合快速验证。

--model-id
    Hugging Face 模型 id。默认继承核心脚本中的 MODEL_ID。

--num-keyframes
    action-aware 关键帧候选密度。默认 32。

--max-vlm-keyframes
    实际送入 Qwen 的最大图像数量。默认 16。

--target-subtasks
    指定固定 sub-task 数量。默认 0，表示根据任务复杂度自动推断。

--max-new-tokens
    Qwen 生成 JSON 的最大 token 数。默认 768。

--max-retries
    JSON 缺失或校验失败时的重试次数。默认 3。

--min-gripper-span-frames
    过滤短暂夹爪命令抖动的最小 span 长度。默认 5。

--min-subtask-frames
    校验 sub-task 是否异常过短的阈值。默认 5。

--disable-action-boundary-prior
    关闭 action/state 边界先验，仅用于对比或调试。

--no-save-keyframes
    不保存关键帧 PNG。默认会保存关键帧。

--overwrite
    覆盖已有 annotation.json。未指定时会跳过已标注 episode。

--stop-on-error
    遇到单个 episode 失败时立即停止。默认记录错误并继续后续 episode。

--summary-jsonl
    自定义 summary JSONL 路径。默认保存到 <OUTPUT_DIR>/summary.jsonl。
```

## 输出结构

输出目录结构如下：

```text
<OUTPUT_DIR>/
  summary.jsonl
  libero_10_no_noops_1.0.0_lerobot/
    episode_000015/
      annotation.json
      raw_model_response.txt
      raw_model_response_attempt_1.txt
      keyframes.json
      keyframes/
        frame_000000.png
        ...
    episode_000016/
      ...
```

如果某个 episode 标注失败，会写入：

```text
<OUTPUT_DIR>/<SUBSET>/episode_XXXXXX/error.json
```

并在 `summary.jsonl` 中记录失败状态。

## annotation.json 主要字段

```text
episode_id
    当前 episode id。

task
    原始语言指令。

dataset_subset
    当前 episode 所属 subset。

source_dataset_dir
    当前 subset 路径。

source_video
    当前 top-view 视频路径。

subtasks
    sub-task 文本和起止帧。

sampled_keyframe_labels
    送入 Qwen 的关键帧与 sub-task 对应关系。

action_boundary_prior
    由 action/state 推导出的边界先验、span 和边界原因。

language_disambiguation
    如存在语言歧义，记录 Qwen 的消歧结果。

frame_labels
    每一帧对应的 sub-task id。
```

## 标注流程摘要

每个 episode 的处理顺序如下：

1. 从 parquet 读取 episode 行数据。
2. 从 `meta/tasks.jsonl` 读取 task 文本。
3. 从 top-view mp4 解码关键帧。
4. 使用 Qwen 对必要的语言歧义进行初始画面判断。
5. 使用 action/state 构造 boundary prior。
6. 基于 boundary prior 选择关键帧。
7. 将 task、关键帧、语言约束和 boundary prior 发送给 Qwen 生成 JSON。
8. 对 JSON 做格式、数量、时序和语义校验。
9. 校验失败时带 feedback 重试。
10. 保存 annotation、raw response、关键帧和 summary。

## 与 Qwen 的交互

脚本中 Qwen 会被调用多次，但职责不同：

- 初始画面消歧：判断代词、空间短语或目标对象归属；
- 主标注：根据关键帧和任务生成 sub-task JSON；
- close 边界 refine：对 drawer/cabinet/microwave/door 等最后关闭动作，从候选帧中判断最后一个 sub-task 的开始。

主标注阶段要求 Qwen 输出纯 JSON。若模型输出分析文本或 JSON 不完整，脚本会提取失败并重试。

## 关键阈值

当前推荐默认值：

```text
--num-keyframes 32
--max-vlm-keyframes 16
--min-gripper-span-frames 5
--min-subtask-frames 5
--max-retries 3
```

额外边界策略：

- 多物体 pick-place 中，短暂 close/open 不会直接视作有效抓取，需要满足最小搬运时长；
- drawer/cabinet/microwave/door 的 close 起点会由 Qwen 判断，但如果选择明显偏晚，会使用 late cap 回退；
- StarVLA/no-noops 的二值夹爪动作会先映射到共享 open/close 定义。

## 进度与恢复

脚本会显示 tqdm 进度条，并在每个 episode 完成后写入 summary。

默认情况下，如果目标 episode 已经存在 `annotation.json`，脚本会跳过该 episode。需要重跑时使用：

```bash
--overwrite
```

如果希望出现错误后立即停止，使用：

```bash
--stop-on-error
```

否则脚本会保存 `error.json` 并继续处理后续 episode。

## 建议工作流

1. 先使用 `--episode-ids` 在少量 episode 上验证。
2. 抽查 `annotation.json`、`keyframes/` 和 `raw_model_response.txt`。
3. 确认质量后再扩大到完整 subset。
4. 完整运行时建议保留关键帧，方便后续抽样检查。
5. 若需要迁移到其他机器，优先通过 `--data-root` 和 `--output-dir` 显式指定路径。
