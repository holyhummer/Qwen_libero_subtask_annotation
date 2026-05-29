import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from annotate_libero_episode_qwen35 import (
    MODEL_ID,
    TOP_VIEW_KEY,
    ambiguous_pickup_clauses,
    ambiguous_pronoun_clauses,
    build_action_boundary_prior,
    build_task_constraints,
    build_validation_feedback,
    choose_action_aware_keyframe_rows,
    extract_json_object,
    generate_annotation,
    infer_target_subtasks,
    normalize_annotation,
    read_parquet,
    resolve_ambiguous_pickup_relations_with_vlm,
    resolve_ambiguous_pronouns_with_vlm,
    resolve_final_close_start_with_vlm,
    save_failed_outputs,
    save_outputs,
    scalar_from_value,
)


DEFAULT_DATA_ROOT = Path(
    "/media/lab1523-2d404-4-1/3b8c86f2-4119-4d3e-8b80-a5abdcfab491/datasets/"
    "starVLA_datasets/libero"
)
DEFAULT_OUTPUT_DIR = Path(
    "/media/lab1523-2d404-4-1/3b8c86f2-4119-4d3e-8b80-a5abdcfab491/datasets/"
    "starVLA_datasets/libero_subtask_annotations"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate StarVLA/native-style LIBERO subsets with the Qwen3.5 VLM strategy."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--subsets",
        default=None,
        help="Comma-separated subset directory names. Defaults to every *_lerobot directory.",
    )
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=None, help="Exclusive local episode id end.")
    parser.add_argument(
        "--episode-ids",
        default=None,
        help="Comma-separated local episode ids to run in each selected subset.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-keyframes", type=int, default=32)
    parser.add_argument("--max-vlm-keyframes", type=int, default=16)
    parser.add_argument("--target-subtasks", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--min-gripper-span-frames", type=int, default=5)
    parser.add_argument("--min-subtask-frames", type=int, default=5)
    parser.add_argument("--disable-action-boundary-prior", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--no-save-keyframes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--summary-jsonl",
        type=Path,
        default=None,
        help="Optional summary JSONL path. Defaults to <output-dir>/summary.jsonl.",
    )
    return parser.parse_args()


def progress_bar(iterable, total):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None, iterable
    return tqdm(iterable, total=total, dynamic_ncols=True, desc="Annotating StarVLA LIBERO"), None


def progress_write(progress, message):
    if progress is None:
        print(message)
    else:
        progress.write(message)


def discover_subset_dirs(data_root, subset_arg):
    if subset_arg:
        subset_names = [name.strip() for name in subset_arg.split(",") if name.strip()]
        subset_dirs = [data_root / name for name in subset_names]
    else:
        subset_dirs = sorted(path for path in data_root.iterdir() if path.is_dir() and path.name.endswith("_lerobot"))

    missing = [str(path) for path in subset_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError("Subset directories do not exist: " + ", ".join(missing))
    if not subset_dirs:
        raise FileNotFoundError(f"No *_lerobot subset directories found under {data_root}")
    return subset_dirs


def load_task_map_jsonl(subset_dir):
    task_map = {}
    task_path = subset_dir / "meta" / "tasks.jsonl"
    with task_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_map[int(row["task_index"])] = str(row["task"])
    return task_map


def load_total_episodes(subset_dir):
    info_path = subset_dir / "meta" / "info.json"
    if not info_path.exists():
        return None
    return int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"])


def parse_episode_ids(args, total_episodes):
    if args.episode_ids:
        episode_ids = [int(value.strip()) for value in args.episode_ids.split(",") if value.strip()]
    else:
        episode_end = args.episode_end
        if episode_end is None:
            if total_episodes is None:
                raise ValueError("--episode-end is required when meta/info.json is unavailable.")
            episode_end = total_episodes
        episode_ids = list(range(args.episode_start, episode_end))

    if args.limit is not None:
        episode_ids = episode_ids[: args.limit]
    return episode_ids


def build_episode_file_index(subset_dir):
    episode_to_file = {}
    for parquet_path in sorted((subset_dir / "data").glob("**/episode_*.parquet")):
        episode_id = int(parquet_path.stem.split("_")[-1])
        episode_to_file[episode_id] = parquet_path
    if not episode_to_file:
        raise FileNotFoundError(f"No episode parquet files found under {subset_dir / 'data'}")
    return episode_to_file


def load_episode_from_index(episode_to_file, episode_id):
    parquet_path = episode_to_file.get(episode_id)
    if parquet_path is None:
        raise ValueError(f"Episode {episode_id} was not found in the subset parquet index.")
    episode_df = read_parquet(parquet_path).copy()
    episode_df["_source_parquet"] = str(parquet_path)
    return episode_df.sort_values("frame_index").reset_index(drop=True)


def episode_df_for_action_prior(episode_df):
    if "action" not in episode_df.columns:
        return episode_df

    try:
        import numpy as np

        actions = np.stack(episode_df["action"].to_numpy())
    except Exception:
        return episode_df

    if actions.ndim != 2 or actions.shape[1] < 1:
        return episode_df

    gripper = actions[:, -1]
    finite = gripper[np.isfinite(gripper)]
    if len(finite) == 0:
        return episode_df

    unique_values = {round(float(value), 6) for value in finite}
    if unique_values.issubset({0.0, 1.0}):
        remapped_actions = actions.copy()
        # StarVLA/no-noops LIBERO stores gripper as 1=open/release and 0=close/hold.
        # The shared action prior expects positive=close/hold and negative=open/release.
        remapped_actions[:, -1] = np.where(gripper >= 0.5, -1.0, 1.0)
        remapped_df = episode_df.copy()
        remapped_df["action"] = list(remapped_actions)
        return remapped_df

    return episode_df


def video_path_for_episode(subset_dir, episode_id, image_key=TOP_VIEW_KEY):
    candidates = sorted((subset_dir / "videos").glob(f"**/{image_key}/episode_{episode_id:06d}.mp4"))
    if not candidates:
        raise FileNotFoundError(
            f"No top-view video found for episode {episode_id} under {subset_dir / 'videos'} "
            f"with image key {image_key}."
        )
    return candidates[0]


def read_video_frames(video_path, frame_indices):
    frame_indices = sorted(set(int(frame) for frame in frame_indices))
    if not frame_indices:
        return {}

    try:
        import av

        selected = {}
        wanted = set(frame_indices)
        max_frame = frame_indices[-1]
        with av.open(str(video_path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index in wanted:
                    selected[frame_index] = Image.fromarray(frame.to_ndarray(format="rgb24"))
                    if len(selected) == len(wanted):
                        break
                if frame_index > max_frame:
                    break
        missing = wanted - set(selected)
        if missing:
            raise RuntimeError(f"Failed to decode frames {sorted(missing)} from {video_path}")
        return selected
    except Exception:
        import imageio.v3 as iio

        return {
            frame_index: Image.fromarray(iio.imread(video_path, index=frame_index)).convert("RGB")
            for frame_index in frame_indices
        }


def attach_images_to_keyframes(keyframes, video_path):
    frame_to_image = read_video_frames(video_path, [frame for frame, _ in keyframes])
    prepared_keyframes = []
    for frame, row in keyframes:
        row = row.copy()
        row["_image"] = frame_to_image[int(frame)].convert("RGB")
        prepared_keyframes.append((frame, row))
    return prepared_keyframes


def close_start_candidate_rows(episode_df, action_boundary_prior, max_candidates=14):
    if not action_boundary_prior:
        return []

    release_reason = next(
        (
            reason
            for reason in action_boundary_prior.get("boundary_reasons", [])
            if reason.get("reason") == "release_start_then_move_to_next_target"
        ),
        None,
    )
    if not release_reason or not release_reason.get("source_span"):
        return []

    start, end = [int(value) for value in release_reason["source_span"]]
    boundary = int(action_boundary_prior["boundary_frames"][-1])
    ratios = (0.15, 0.22, 0.30, 0.38, 0.45, 0.52, 0.60, 0.70, 0.85, 1.0)
    candidates = {start, boundary}
    for ratio in ratios:
        candidates.add(round(start + ratio * (end - start)))

    frame_values = [int(frame) for frame in episode_df["frame_index"].to_list()]
    frame_to_row = {int(row["frame_index"]): row for _, row in episode_df.iterrows()}
    selected = []
    for candidate in sorted(candidates):
        nearest = min(frame_values, key=lambda frame: abs(frame - candidate))
        if start <= nearest <= end and nearest not in selected:
            selected.append(nearest)
    selected = selected[:max_candidates]
    return [(frame, frame_to_row[frame]) for frame in selected]


def prepare_keyframes(episode_df, video_path, num_keyframes, max_vlm_keyframes, action_boundary_prior):
    effective_keyframes = min(num_keyframes, max_vlm_keyframes)
    keyframes = choose_action_aware_keyframe_rows(
        episode_df,
        effective_keyframes,
        boundary_frames=(action_boundary_prior or {}).get("boundary_frames"),
    )
    return attach_images_to_keyframes(keyframes, video_path)


def write_summary(summary_path, record):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_error(output_dir, subset_name, episode_id, error, raw_attempts=None):
    subset_output_dir = output_dir / subset_name
    episode_dir = subset_output_dir / f"episode_{episode_id:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    error_path = episode_dir / "error.json"
    error_path.write_text(
        json.dumps(
            {
                "subset": subset_name,
                "episode_id": episode_id,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if raw_attempts:
        save_failed_outputs(subset_output_dir, episode_id, raw_attempts)
    return error_path


def annotate_episode(
    *,
    subset_dir,
    subset_name,
    episode_id,
    episode_df,
    video_path,
    task,
    model,
    processor,
    args,
):
    constraints = build_task_constraints(task)
    if ambiguous_pickup_clauses(constraints) or ambiguous_pronoun_clauses(constraints):
        initial_image = read_video_frames(video_path, [int(episode_df.iloc[0]["frame_index"])])[
            int(episode_df.iloc[0]["frame_index"])
        ]
    if ambiguous_pronoun_clauses(constraints):
        constraints = resolve_ambiguous_pronouns_with_vlm(
            model=model,
            processor=processor,
            task=task,
            constraints=constraints,
            initial_image=initial_image,
            enable_thinking=args.enable_thinking,
        )
    if ambiguous_pickup_clauses(constraints):
        constraints = resolve_ambiguous_pickup_relations_with_vlm(
            model=model,
            processor=processor,
            task=task,
            constraints=constraints,
            initial_image=initial_image,
            enable_thinking=args.enable_thinking,
        )

    target_subtasks = args.target_subtasks
    if target_subtasks <= 0:
        target_subtasks = infer_target_subtasks(constraints)

    prior_episode_df = episode_df_for_action_prior(episode_df)
    preliminary_action_boundary_prior = None
    if not args.disable_action_boundary_prior:
        preliminary_action_boundary_prior = build_action_boundary_prior(
            episode_df=prior_episode_df,
            constraints=constraints,
            target_subtasks=target_subtasks,
            keyframe_frames=[],
            min_gripper_span_frames=args.min_gripper_span_frames,
        )

    prepared_keyframes = prepare_keyframes(
        episode_df=episode_df,
        video_path=video_path,
        num_keyframes=args.num_keyframes,
        max_vlm_keyframes=args.max_vlm_keyframes,
        action_boundary_prior=preliminary_action_boundary_prior,
    )

    action_boundary_prior = None
    if not args.disable_action_boundary_prior:
        action_boundary_prior = build_action_boundary_prior(
            episode_df=prior_episode_df,
            constraints=constraints,
            target_subtasks=target_subtasks,
            keyframe_frames=[frame for frame, _ in prepared_keyframes],
            min_gripper_span_frames=args.min_gripper_span_frames,
        )
        close_candidate_rows = close_start_candidate_rows(episode_df, action_boundary_prior)
        if close_candidate_rows:
            close_candidate_keyframes = attach_images_to_keyframes(close_candidate_rows, video_path)
            action_boundary_prior = resolve_final_close_start_with_vlm(
                model=model,
                processor=processor,
                task=task,
                constraints=constraints,
                action_boundary_prior=action_boundary_prior,
                candidate_keyframes=close_candidate_keyframes,
                enable_thinking=args.enable_thinking,
            )

    raw_attempts = []
    retry_feedback = None
    annotation = None
    validation_feedback = []
    raw_response = ""

    for _ in range(args.max_retries + 1):
        raw_response = generate_annotation(
            model=model,
            processor=processor,
            task=task,
            keyframes=prepared_keyframes,
            target_subtasks=target_subtasks,
            constraints=constraints,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            action_boundary_prior=action_boundary_prior,
            retry_feedback=retry_feedback,
        )
        raw_attempts.append(raw_response)

        try:
            raw_annotation = extract_json_object(raw_response)
        except ValueError:
            retry_feedback = (
                "The previous response did not contain any JSON object. "
                "Output only one JSON object that starts with { and ends with }."
            )
            validation_feedback = [retry_feedback]
            continue

        annotation = normalize_annotation(
            raw_annotation=raw_annotation,
            episode_id=episode_id,
            task=task,
            first_frame=int(episode_df["frame_index"].min()),
            last_frame=int(episode_df["frame_index"].max()),
            keyframe_frames=[frame for frame, _ in prepared_keyframes],
            action_boundary_prior=action_boundary_prior,
        )
        annotation["dataset_subset"] = subset_name
        annotation["source_dataset_dir"] = str(subset_dir)
        annotation["source_video"] = str(video_path)
        if constraints.get("language_disambiguation"):
            annotation["language_disambiguation"] = constraints["language_disambiguation"]

        validation_feedback = build_validation_feedback(
            annotation,
            constraints,
            target_subtasks,
            min_subtask_frames=args.min_subtask_frames,
        )
        if not validation_feedback:
            break

        retry_feedback = "\n".join(f"- {issue}" for issue in validation_feedback)

    if annotation is None:
        raise RuntimeError("Failed to produce a JSON annotation after retries.")

    subset_output_dir = args.output_dir / subset_name
    annotation_path = save_outputs(
        output_dir=subset_output_dir,
        annotation=annotation,
        raw_response=raw_response,
        keyframes=prepared_keyframes,
        save_keyframes=not args.no_save_keyframes,
        raw_attempts=raw_attempts,
    )

    return {
        "subset": subset_name,
        "episode_id": episode_id,
        "task": task,
        "status": "ok" if not validation_feedback else "warning",
        "annotation_path": str(annotation_path),
        "target_subtasks": target_subtasks,
        "subtasks": len(annotation["subtasks"]),
        "keyframes": len(prepared_keyframes),
        "validation_feedback": validation_feedback,
    }


def main():
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    if args.num_keyframes < 2:
        raise ValueError("--num-keyframes must be at least 2")
    if args.max_vlm_keyframes < 2:
        raise ValueError("--max-vlm-keyframes must be at least 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_jsonl or (args.output_dir / "summary.jsonl")

    subset_dirs = discover_subset_dirs(args.data_root, args.subsets)
    jobs = []
    subset_context = {}
    for subset_dir in subset_dirs:
        total_episodes = load_total_episodes(subset_dir)
        episode_ids = parse_episode_ids(args, total_episodes)
        subset_context[subset_dir.name] = {
            "dir": subset_dir,
            "task_map": load_task_map_jsonl(subset_dir),
            "episode_to_file": build_episode_file_index(subset_dir),
        }
        for episode_id in episode_ids:
            jobs.append((subset_dir.name, episode_id))

    print(f"Data root: {args.data_root}")
    print(f"Subsets: {', '.join(subset_context)}")
    print(f"Episodes requested: {len(jobs)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Summary JSONL: {summary_path}")

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    counts = {"ok": 0, "warning": 0, "failed": 0, "skipped": 0}
    started_at = time.time()
    progress, fallback_iterable = progress_bar(jobs, len(jobs))
    job_iterable = progress if progress is not None else fallback_iterable

    for position, (subset_name, episode_id) in enumerate(job_iterable, start=1):
        annotation_path = args.output_dir / subset_name / f"episode_{episode_id:06d}" / "annotation.json"
        if annotation_path.exists() and not args.overwrite:
            record = {
                "subset": subset_name,
                "episode_id": episode_id,
                "status": "skipped",
                "annotation_path": str(annotation_path),
            }
            counts["skipped"] += 1
            write_summary(summary_path, record)
            progress_write(progress, f"[{position}/{len(jobs)}] {subset_name} episode={episode_id} skipped")
            if progress is not None:
                progress.set_postfix(counts)
            continue

        context = subset_context[subset_name]
        try:
            episode_df = load_episode_from_index(context["episode_to_file"], episode_id)
            task_index = int(scalar_from_value(episode_df.iloc[0]["task_index"]))
            task = context["task_map"][task_index]
            video_path = video_path_for_episode(context["dir"], episode_id, TOP_VIEW_KEY)
            record = annotate_episode(
                subset_dir=context["dir"],
                subset_name=subset_name,
                episode_id=episode_id,
                episode_df=episode_df,
                video_path=video_path,
                task=task,
                model=model,
                processor=processor,
                args=args,
            )
            counts[record["status"]] += 1
            write_summary(summary_path, record)
            progress_write(
                progress,
                f"[{position}/{len(jobs)}] {subset_name} episode={episode_id} "
                f"status={record['status']} subtasks={record['subtasks']} "
                f"keyframes={record['keyframes']}",
            )
            if progress is not None:
                progress.set_postfix(counts)
        except Exception as exc:
            counts["failed"] += 1
            error_path = write_error(args.output_dir, subset_name, episode_id, exc)
            record = {
                "subset": subset_name,
                "episode_id": episode_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "error_path": str(error_path),
            }
            write_summary(summary_path, record)
            progress_write(progress, f"[{position}/{len(jobs)}] {subset_name} episode={episode_id} failed: {exc}")
            if progress is not None:
                progress.set_postfix(counts)
            if args.stop_on_error:
                if progress is not None:
                    progress.close()
                raise

    if progress is not None:
        progress.close()

    elapsed = time.time() - started_at
    print(f"Done in {elapsed / 60:.1f} min")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
