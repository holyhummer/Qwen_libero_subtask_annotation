# StarVLA/native LIBERO Dataset Annotation Script

This document describes `tools/annotate_starvla_libero_dataset_qwen35.py`, including its purpose, expected dataset layout, usage, output format, and recovery behavior. Paths in this document are written as placeholders. Pass the actual paths on your machine through command-line arguments.

## Purpose

`tools/annotate_starvla_libero_dataset_qwen35.py` performs batch sub-task annotation for StarVLA/native-style LIBERO datasets. It reuses the core single-episode annotation logic, including:

- Qwen VLM semantic understanding and JSON annotation generation;
- initial-frame language disambiguation;
- action/state-based boundary priors;
- action-aware keyframe selection;
- special close-action boundary refinement;
- validation, retries, error logging, and progress display.

The script is designed for full-dataset annotation. Each episode is saved in its own annotation directory.

## Input Dataset Layout

The script expects `--data-root` to contain one or more subset directories ending with `_lerobot`, for example:

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

The current annotation pipeline only uses the top-view camera:

```text
observation.images.image
```

The wrist/hand camera is not sent to Qwen.

## Environment

Create a dedicated conda environment for the annotation script:

```bash
conda create -n libero_annot_qwen35 python=3.12 -y
conda activate libero_annot_qwen35
```

Install PyTorch according to the CUDA version on your machine. For example, for CUDA 12.6:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

For CPU-only usage or another CUDA version, replace this with the matching PyTorch installation command.

Install the remaining packages used by the annotation scripts:

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

One validated package set is:

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

After installation, check the script arguments:

```bash
conda run -n libero_annot_qwen35 python tools/annotate_starvla_libero_dataset_qwen35.py --help
```

## Quick Start

Annotate one episode from `libero_10`:

```bash
conda run -n libero_annot_qwen35 python tools/annotate_starvla_libero_dataset_qwen35.py \
  --data-root <DATA_ROOT> \
  --output-dir <OUTPUT_DIR> \
  --subsets libero_10_no_noops_1.0.0_lerobot \
  --episode-ids 15 \
  --overwrite
```

Annotate the first 20 episodes of one subset:

```bash
conda run -n libero_annot_qwen35 python tools/annotate_starvla_libero_dataset_qwen35.py \
  --data-root <DATA_ROOT> \
  --output-dir <OUTPUT_DIR> \
  --subsets libero_10_no_noops_1.0.0_lerobot \
  --episode-start 0 \
  --episode-end 20 \
  --overwrite
```

Annotate all `_lerobot` subsets under `--data-root`:

```bash
conda run -n libero_annot_qwen35 python tools/annotate_starvla_libero_dataset_qwen35.py \
  --data-root <DATA_ROOT> \
  --output-dir <OUTPUT_DIR>
```

For a small trial run, use:

```bash
--limit 10
```

## Common Arguments

```text
--data-root
    Root directory of the StarVLA/native LIBERO dataset. Passing this explicitly is recommended.

--output-dir
    Annotation output directory. Passing this explicitly is recommended.

--subsets
    Comma-separated subset directory names. If omitted, the script scans all *_lerobot directories under --data-root.

--episode-ids
    Comma-separated episode ids, for example 0,1,2,15.

--episode-start / --episode-end
    Episode range. The end value is exclusive.

--limit
    Truncate the selected episode list. Useful for quick validation.

--model-id
    Hugging Face model id. Defaults to the MODEL_ID defined by the core annotation script.

--num-keyframes
    Candidate density for action-aware keyframe selection. Default: 32.

--max-vlm-keyframes
    Maximum number of images sent to Qwen. Default: 16.

--target-subtasks
    Fixed number of sub-tasks. Default: 0, which means automatic inference from task complexity.

--max-new-tokens
    Maximum generation budget for Qwen JSON output. Default: 768.

--max-retries
    Retry count when JSON is missing or validation fails. Default: 3.

--min-gripper-span-frames
    Minimum span length for filtering short gripper-command glitches. Default: 5.

--min-subtask-frames
    Validation threshold for suspiciously short sub-tasks. Default: 5.

--disable-action-boundary-prior
    Disable action/state boundary priors. Mainly useful for ablation or debugging.

--no-save-keyframes
    Do not save keyframe PNGs. By default, keyframes are saved.

--overwrite
    Overwrite existing annotation.json files. Without this flag, already annotated episodes are skipped.

--stop-on-error
    Stop immediately when one episode fails. By default, the script records the error and continues.

--summary-jsonl
    Custom summary JSONL path. Defaults to <OUTPUT_DIR>/summary.jsonl.
```

## Output Layout

The output directory is organized as follows:

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

If an episode fails, the script writes:

```text
<OUTPUT_DIR>/<SUBSET>/episode_XXXXXX/error.json
```

The failure is also recorded in `summary.jsonl`.

## Main annotation.json Fields

```text
episode_id
    Episode id.

task
    Original language instruction.

dataset_subset
    Dataset subset name.

source_dataset_dir
    Source subset directory.

source_video
    Source top-view video path.

subtasks
    Sub-task text and frame ranges.

sampled_keyframe_labels
    Mapping between keyframes sent to Qwen and sub-task ids.

action_boundary_prior
    Boundary priors, action spans, and boundary reasons derived from action/state signals.

language_disambiguation
    Qwen disambiguation result, when language ambiguity exists.

frame_labels
    Per-frame sub-task labels.
```

## Annotation Flow

For each episode, the script performs the following steps:

1. Read episode rows from parquet files.
2. Read the task text from `meta/tasks.jsonl`.
3. Decode keyframes from the top-view mp4.
4. Use Qwen to resolve language ambiguity from the initial image when needed.
5. Build an action/state-based boundary prior.
6. Select keyframes using the boundary prior.
7. Send the task, keyframes, language constraints, and boundary prior to Qwen to generate JSON.
8. Validate JSON format, sub-task count, temporal order, and semantic consistency.
9. Retry with feedback if validation fails.
10. Save the annotation, raw responses, keyframes, and summary record.

## Qwen Interaction

The script may call Qwen multiple times for different purposes:

- Initial-frame disambiguation: resolves pronouns, spatial phrases, and target-object ambiguity.
- Main annotation: generates sub-task JSON from keyframes and the task instruction.
- Close-boundary refinement: for final close actions on drawers, cabinets, microwaves, or doors, selects the start frame of the last sub-task from candidate frames.

The main annotation prompt requires pure JSON output. If the model emits analysis text or incomplete JSON, the script treats it as a failed attempt and retries.

## Key Thresholds

Recommended defaults:

```text
--num-keyframes 32
--max-vlm-keyframes 16
--min-gripper-span-frames 5
--min-subtask-frames 5
--max-retries 3
```

Additional boundary policies:

- In multi-object pick-place tasks, short close/open segments are not treated as successful grasps unless there is enough movement after the grasp.
- For drawer/cabinet/microwave/door close actions, Qwen selects the close-start candidate frame. If the selected frame is visibly too late, a late-cap correction moves the boundary earlier.
- Binary gripper actions in StarVLA/no-noops data are remapped to the shared open/close convention before building action priors.

## Progress and Recovery

The script displays a tqdm progress bar and writes a summary record after each episode.

By default, if an episode already has `annotation.json`, it is skipped. To rerun existing episodes, use:

```bash
--overwrite
```

To stop on the first failure, use:

```bash
--stop-on-error
```

Otherwise, the script saves `error.json` and continues with later episodes.

## Recommended Workflow

1. Validate on a small set of episodes with `--episode-ids`.
2. Inspect `annotation.json`, `keyframes/`, and `raw_model_response.txt`.
3. Scale up to a full subset after the sample quality is acceptable.
4. Keep keyframes during full runs to support later sampling checks.
5. For portability, always pass `--data-root` and `--output-dir` explicitly.
