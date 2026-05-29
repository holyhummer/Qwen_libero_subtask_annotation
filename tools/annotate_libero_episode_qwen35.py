import argparse
import json
import re
from io import BytesIO
from pathlib import Path

from PIL import Image


DATASET_ID = "HuggingFaceVLA/libero"
MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_LOCAL_DATA_DIR = Path(
    "/home/lab1523-2d404-4-1/.cache/huggingface/lerobot/HuggingFaceVLA/libero"
)
TOP_VIEW_KEY = "observation.images.image"
COLOR_WORDS = (
    "black",
    "blue",
    "brown",
    "chocolate",
    "cream",
    "green",
    "grey",
    "gray",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
)
OBJECT_WORDS = (
    "basket",
    "book",
    "bowl",
    "box",
    "cabinet",
    "caddy",
    "cheese",
    "drawer",
    "microwave",
    "moka pot",
    "mug",
    "plate",
    "pudding",
    "rack",
    "soup",
    "sauce",
    "stove",
    "bottle",
)
PREPOSITIONS = (
    " on ",
    " onto ",
    " in ",
    " into ",
    " inside ",
    " to ",
    " above ",
    " under ",
    " at ",
)
TRAILING_PREPOSITIONS = ("inside", "in", "on", "onto", "to", "into")
CONTROL_ACTIONS = {"close", "open", "turn on", "turn off"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate one LIBERO episode with Qwen3.5 VLM sub-task time segments."
    )
    parser.add_argument("--episode-id", type=int, default=8)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("libero_subtask_annotations"))
    parser.add_argument("--num-keyframes", type=int, default=16)
    parser.add_argument(
        "--max-vlm-keyframes",
        type=int,
        default=16,
        help="Maximum keyframes sent to the VLM. Higher candidate counts are down-selected.",
    )
    parser.add_argument(
        "--target-subtasks",
        type=int,
        default=0,
        help="Preferred number of semantic sub-tasks. Use 0 to infer from task complexity.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--min-gripper-span-frames",
        type=int,
        default=5,
        help="Merge gripper command glitches shorter than this many frames before making boundary priors.",
    )
    parser.add_argument(
        "--min-subtask-frames",
        type=int,
        default=5,
        help="Validation threshold for suspiciously short generated sub-task segments.",
    )
    parser.add_argument(
        "--disable-action-boundary-prior",
        action="store_true",
        help="Do not use action gripper-command transitions as boundary priors.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow models with thinking mode to emit reasoning. Disabled by default for JSON annotation.",
    )
    parser.add_argument(
        "--save-keyframes",
        action="store_true",
        help="Save the sampled top-view keyframes beside the JSON output.",
    )
    return parser.parse_args()


def read_parquet(path, columns=None):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Please install pandas and pyarrow to read local LIBERO parquet files.") from exc

    try:
        return pd.read_parquet(path, columns=columns)
    except OSError as exc:
        raise OSError(
            f"Failed to read {path}. Try upgrading pyarrow, for example: pip install -U pyarrow"
        ) from exc


def scalar_from_value(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return scalar_from_value(value[0])
    return value


def image_from_value(value, dataset_dir=None):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), bytes):
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            image_path = Path(value["path"])
            if not image_path.is_absolute() and dataset_dir is not None:
                image_path = dataset_dir / image_path
            return Image.open(image_path).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)}")


def load_task_map(local_data_dir):
    tasks_df = read_parquet(local_data_dir / "meta" / "tasks.parquet")
    task_map = {}
    for task_text, row in tasks_df.iterrows():
        task_map[int(row["task_index"])] = str(task_text)
    return task_map


def load_episode(local_data_dir, episode_id):
    parquet_files = sorted((local_data_dir / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {local_data_dir / 'data'}")

    episode_parts = []
    for parquet_path in parquet_files:
        meta_df = read_parquet(
            parquet_path,
            columns=["episode_index", "frame_index", "task_index", "timestamp"],
        )
        if not (meta_df["episode_index"] == episode_id).any():
            continue

        df = read_parquet(parquet_path)
        df = df[df["episode_index"] == episode_id].copy()
        df["_source_parquet"] = str(parquet_path)
        episode_parts.append(df)

    if not episode_parts:
        raise ValueError(f"Episode {episode_id} was not found under {local_data_dir}")

    import pandas as pd

    episode_df = pd.concat(episode_parts, ignore_index=True)
    return episode_df.sort_values("frame_index").reset_index(drop=True)


def choose_keyframe_rows(episode_df, num_keyframes):
    frame_count = len(episode_df)
    if frame_count == 0:
        raise ValueError("Cannot choose keyframes from an empty episode.")
    if num_keyframes <= 0:
        return []
    if num_keyframes == 1:
        index = round((frame_count - 1) / 2)
        return [(int(episode_df.iloc[index]["frame_index"]), episode_df.iloc[index])]
    if num_keyframes >= frame_count:
        indices = list(range(frame_count))
    else:
        indices = sorted(
            {
                round(i * (frame_count - 1) / (num_keyframes - 1))
                for i in range(num_keyframes)
            }
        )
    return [(int(episode_df.iloc[i]["frame_index"]), episode_df.iloc[i]) for i in indices]


def choose_action_aware_keyframe_rows(episode_df, num_keyframes, boundary_frames=None):
    if len(episode_df) == 0:
        raise ValueError("Cannot choose keyframes from an empty episode.")
    if num_keyframes <= 0:
        return []

    frame_values = [int(frame) for frame in episode_df["frame_index"].to_list()]
    frame_set = set(frame_values)
    selected = set()
    boundary_frames = sorted(set(boundary_frames or []))

    for boundary in boundary_frames:
        for candidate in (boundary - 1, boundary, boundary + 1):
            if candidate in frame_set:
                selected.add(candidate)

    anchors = [frame_values[0], *boundary_frames, frame_values[-1]]
    anchors = sorted(set(frame for frame in anchors if frame in frame_set))
    for start, end in zip(anchors, anchors[1:]):
        for ratio in (0.5, 0.75):
            candidate = round(start + (end - start) * ratio)
            nearest = min(frame_values, key=lambda frame: abs(frame - candidate))
            selected.add(nearest)
    selected.add(frame_values[0])
    selected.add(frame_values[-1])

    remaining = max(0, num_keyframes - len(selected))
    if remaining > 0:
        for frame, _ in choose_keyframe_rows(episode_df, remaining):
            selected.add(frame)

    selected = sorted(selected)
    if len(selected) > num_keyframes:
        keep = {
            frame
            for frame in (frame_values[0], frame_values[-1], *boundary_frames)
            if frame in frame_set
        }
        while len(keep) < num_keyframes:
            candidates = [frame for frame in selected if frame not in keep]
            if not candidates:
                break
            next_frame = max(
                candidates,
                key=lambda frame: min(abs(frame - kept) for kept in keep),
            )
            keep.add(next_frame)
        selected = sorted(keep)

    frame_to_row = {int(row["frame_index"]): row for _, row in episode_df.iterrows()}
    return [(frame, frame_to_row[frame]) for frame in selected]


def build_action_boundary_prior(
    episode_df,
    constraints,
    target_subtasks,
    keyframe_frames,
    min_gripper_span_frames=5,
):
    pick_place_clauses = [
        clause
        for clause in constraints["clauses"]
        if clause.get("action") in {"put", "place", "pick up", "push"}
        and clause.get("object")
        and clause.get("target")
    ]
    close_open_clauses = [
        clause
        for clause in constraints["clauses"]
        if clause.get("action") in CONTROL_ACTIONS and clause.get("object")
    ]
    supports_pick_place_prior = target_subtasks == len(pick_place_clauses) * 2
    supports_put_close_prior = (
        len(pick_place_clauses) == 1
        and len(close_open_clauses) == 1
        and target_subtasks == 5
    )
    supports_control_then_pick_place_prior = (
        len(pick_place_clauses) == 1
        and len(close_open_clauses) == 1
        and constraints["clauses"].index(close_open_clauses[0])
        < constraints["clauses"].index(pick_place_clauses[0])
        and target_subtasks == 3
    )
    if "action" not in episode_df.columns or not (
        supports_pick_place_prior
        or supports_put_close_prior
        or supports_control_then_pick_place_prior
    ):
        return None

    try:
        import numpy as np

        actions = np.stack(episode_df["action"].to_numpy())
    except Exception:
        return None

    if actions.ndim != 2 or actions.shape[1] < 1:
        return None

    frames = [int(frame) for frame in episode_df["frame_index"].to_list()]
    gripper_cmd = actions[:, -1]
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in gripper_cmd]
    frame_to_index = {frame: idx for idx, frame in enumerate(frames)}

    finger_opening = None
    if "observation.state" in episode_df.columns:
        try:
            states = np.stack(episode_df["observation.state"].to_numpy())
            if states.ndim == 2 and states.shape[1] >= 2:
                finger_opening = np.abs(states[:, -2] - states[:, -1])
        except Exception:
            finger_opening = None

    spans = []
    start_idx = 0
    for idx in range(1, len(signs)):
        if signs[idx] != signs[start_idx]:
            spans.append(
                {
                    "start_index": start_idx,
                    "end_index": idx - 1,
                    "start_frame": frames[start_idx],
                    "end_frame": frames[idx - 1],
                    "gripper_command": "close_or_hold" if signs[start_idx] > 0 else "open_or_release",
                }
            )
            start_idx = idx
    spans.append(
        {
            "start_index": start_idx,
            "end_index": len(signs) - 1,
            "start_frame": frames[start_idx],
            "end_frame": frames[-1],
            "gripper_command": "close_or_hold" if signs[start_idx] > 0 else "open_or_release",
        }
    )
    spans = merge_short_gripper_glitches(spans, frames, min_gripper_span_frames)

    if len(spans) < 2:
        return None

    def grasp_stable_frame(close_span):
        if finger_opening is None:
            return close_span["start_frame"]

        start = close_span["start_index"]
        end = close_span["end_index"]
        if end <= start:
            return close_span["start_frame"]

        span_opening = finger_opening[start : end + 1]
        start_opening = float(span_opening[0])
        min_opening = float(span_opening.min())
        closing_range = max(0.0, start_opening - min_opening)
        if closing_range <= 1e-6:
            return close_span["start_frame"]

        plateau_opening = min_opening + 0.4 * closing_range
        strict_opening = min_opening + 0.15 * closing_range
        stable_delta = max(0.0015, 0.03 * closing_range)
        window = 5

        for idx in range(start + 1, end + 1):
            local_end = min(end + 1, idx + window)
            local_delta = np.abs(np.diff(finger_opening[idx - 1 : local_end])).max(initial=0.0)
            if float(finger_opening[idx]) <= plateau_opening and float(local_delta) <= stable_delta:
                return frames[idx]

        for idx in range(start + 1, end + 1):
            local_end = min(end + 1, idx + 3)
            local_delta = np.abs(np.diff(finger_opening[idx - 1 : local_end])).max(initial=0.0)
            if float(finger_opening[idx]) <= strict_opening and float(local_delta) <= stable_delta:
                return frames[idx]

        min_idx = int(start + span_opening.argmin())
        return frames[min_idx]

    object_count = len(pick_place_clauses)
    close_spans = [span for span in spans if span["gripper_command"] == "close_or_hold"]
    boundary_frames = [frames[0]]
    boundary_reasons = [{"frame": frames[0], "reason": "episode_start"}]
    if supports_control_then_pick_place_prior and close_spans:
        first_control_span = close_spans[0]
        post_control_open_span = next(
            (
                span
                for span in spans
                if span["gripper_command"] == "open_or_release"
                and span["start_frame"] > first_control_span["start_frame"]
            ),
            None,
        )
        object_close_span = next(
            (
                span
                for span in close_spans[1:]
                if post_control_open_span is None
                or span["start_frame"] > post_control_open_span["start_frame"]
            ),
            None,
        )
        if post_control_open_span is not None:
            boundary_frames.append(post_control_open_span["start_frame"])
            boundary_reasons.append(
                {
                    "frame": post_control_open_span["start_frame"],
                    "reason": "control_action_complete_then_move_to_object",
                    "source_span": [
                        post_control_open_span["start_frame"],
                        post_control_open_span["end_frame"],
                    ],
                }
            )
        if object_close_span is not None:
            stable_frame = grasp_stable_frame(object_close_span)
            boundary_frames.append(stable_frame)
            boundary_reasons.append(
                {
                    "frame": stable_frame,
                    "reason": "grasp_stable",
                    "source_span": [
                        object_close_span["start_frame"],
                        object_close_span["end_frame"],
                    ],
                }
            )
    elif supports_put_close_prior and close_spans:
        close_span = close_spans[0]
        release_span = next(
            (
                span
                for span in spans
                if span["gripper_command"] == "open_or_release"
                and span["start_frame"] > close_span["start_frame"]
            ),
            None,
        )
        stable_frame = grasp_stable_frame(close_span)
        boundary_frames.extend([close_span["start_frame"], stable_frame])
        boundary_reasons.extend(
            [
                {
                    "frame": close_span["start_frame"],
                    "reason": "grasp_start",
                    "source_span": [close_span["start_frame"], close_span["end_frame"]],
                },
                {
                    "frame": stable_frame,
                    "reason": "grasp_stable",
                    "source_span": [close_span["start_frame"], close_span["end_frame"]],
                },
            ]
        )
        if release_span is not None:
            boundary_frames.append(release_span["start_frame"])
            boundary_reasons.append(
                {
                    "frame": release_span["start_frame"],
                    "reason": "release_start_then_move_to_next_target",
                    "source_span": [release_span["start_frame"], release_span["end_frame"]],
                }
            )
            close_object = normalize_phrase(close_open_clauses[0].get("object", ""))
            close_start_ratio = 0.55 if re.search(r"\b(drawer|cabinet)\b", close_object) else 0.75
            estimated_close_start = round(
                release_span["start_frame"]
                + close_start_ratio * (release_span["end_frame"] - release_span["start_frame"])
            )
            estimated_close_start = min(
                frames,
                key=lambda frame: abs(frame - estimated_close_start),
            )
            boundary_frames.append(estimated_close_start)
            boundary_reasons.append(
                {
                    "frame": estimated_close_start,
                    "reason": "estimated_open_close_start_after_reposition",
                    "close_start_ratio": close_start_ratio,
                    "source_span": [release_span["start_frame"], release_span["end_frame"]],
                }
            )
    elif len(close_spans) >= object_count:
        selected_close_spans = select_pick_place_close_spans(
            close_spans,
            spans,
            frames[-1],
            grasp_stable_frame,
            object_count,
            min_gripper_span_frames,
        )
        for object_idx, close_span in enumerate(selected_close_spans):
            stable_frame = grasp_stable_frame(close_span)
            boundary_frames.append(stable_frame)
            boundary_reasons.append(
                {
                    "frame": stable_frame,
                    "reason": "grasp_stable",
                    "source_span": [close_span["start_frame"], close_span["end_frame"]],
                }
            )

            if object_idx < object_count - 1:
                next_open_span = next(
                    (
                        span
                        for span in spans
                        if span["gripper_command"] == "open_or_release"
                        and span["start_frame"] > close_span["start_frame"]
                    ),
                    None,
                )
                if next_open_span is not None:
                    boundary_frames.append(next_open_span["start_frame"])
                    boundary_reasons.append(
                        {
                            "frame": next_open_span["start_frame"],
                            "reason": "release_start",
                            "source_span": [
                                next_open_span["start_frame"],
                                next_open_span["end_frame"],
                            ],
                        }
                    )

    if (
        not supports_put_close_prior
        and not supports_control_then_pick_place_prior
        and len(boundary_frames) < target_subtasks
    ):
        boundary_frames = [frames[0]]
        boundary_reasons = [{"frame": frames[0], "reason": "episode_start"}]
        for span in spans[1:target_subtasks]:
            boundary_frames.append(span["start_frame"])
            boundary_reasons.append(
                {
                    "frame": span["start_frame"],
                    "reason": "gripper_command_transition",
                    "source_span": [span["start_frame"], span["end_frame"]],
                }
            )

    boundary_frames = boundary_frames[:target_subtasks]
    boundary_reasons = boundary_reasons[:target_subtasks]

    sampled_labels = []
    for frame in keyframe_frames:
        subtask_idx = 0
        for idx, boundary in enumerate(boundary_frames):
            if frame >= boundary:
                subtask_idx = idx
        sampled_labels.append({"frame": frame, "subtask_id": f"S{subtask_idx + 1}"})

    return {
        "spans": [
            {
                "start_frame": span["start_frame"],
                "end_frame": span["end_frame"],
                "gripper_command": span["gripper_command"],
            }
            for span in spans
        ],
        "boundary_frames": boundary_frames,
        "boundary_reasons": boundary_reasons,
        "sampled_keyframe_labels": sampled_labels,
    }


def merge_short_gripper_glitches(spans, frames, min_span_frames):
    if min_span_frames <= 1:
        return spans

    def span_len(span):
        return span["end_index"] - span["start_index"] + 1

    def merged_span(start_span, end_span, gripper_command):
        return {
            "start_index": start_span["start_index"],
            "end_index": end_span["end_index"],
            "start_frame": frames[start_span["start_index"]],
            "end_frame": frames[end_span["end_index"]],
            "gripper_command": gripper_command,
        }

    # Remove very short command islands between two spans with the same command.
    # This keeps real release/reposition phases while ignoring one-off gripper jitters.
    changed = True
    while changed and len(spans) >= 3:
        changed = False
        merged = []
        idx = 0
        while idx < len(spans):
            if (
                idx + 2 < len(spans)
                and span_len(spans[idx + 1]) < min_span_frames
                and spans[idx]["gripper_command"] == spans[idx + 2]["gripper_command"]
            ):
                merged.append(
                    merged_span(
                        spans[idx],
                        spans[idx + 2],
                        spans[idx]["gripper_command"],
                    )
                )
                idx += 3
                changed = True
            else:
                merged.append(spans[idx])
                idx += 1
        spans = merged

    return spans


def select_pick_place_close_spans(
    close_spans,
    spans,
    last_frame,
    grasp_stable_frame,
    object_count,
    min_move_after_grasp_frames,
):
    successful_spans = []
    min_successful_move_frames = max(
        min_move_after_grasp_frames,
        round((last_frame + 1) * 0.05),
    )
    for close_span in close_spans:
        stable_frame = grasp_stable_frame(close_span)
        next_open_span = next(
            (
                span
                for span in spans
                if span["gripper_command"] == "open_or_release"
                and span["start_frame"] > close_span["start_frame"]
            ),
            None,
        )
        if next_open_span is None:
            move_after_grasp_frames = last_frame - stable_frame + 1
        else:
            move_after_grasp_frames = next_open_span["start_frame"] - stable_frame

        if move_after_grasp_frames >= min_successful_move_frames:
            successful_spans.append(close_span)

    if len(successful_spans) >= object_count:
        return successful_spans[:object_count]

    return close_spans[:object_count]


def task_complexity_counts(constraints):
    pick_place_clauses = [
        clause
        for clause in constraints["clauses"]
        if clause.get("action") in {"put", "place", "pick up", "push"}
        and clause.get("object")
        and clause.get("target")
    ]
    close_open_clauses = [
        clause
        for clause in constraints["clauses"]
        if clause.get("action") in CONTROL_ACTIONS and clause.get("object")
    ]
    return pick_place_clauses, close_open_clauses


def infer_target_subtasks(constraints):
    pick_place_clauses, close_open_clauses = task_complexity_counts(constraints)
    if len(pick_place_clauses) == 1 and len(close_open_clauses) == 1:
        control_index = constraints["clauses"].index(close_open_clauses[0])
        pick_index = constraints["clauses"].index(pick_place_clauses[0])
        if control_index < pick_index:
            return 3
        return 5
    if close_open_clauses:
        return max(1, len(pick_place_clauses) * 2 + len(close_open_clauses))
    if pick_place_clauses:
        return max(1, len(pick_place_clauses) * 2)
    return max(1, len(constraints["clauses"]))


def normalize_phrase(text):
    text = str(text or "")
    text = text.lower().strip()
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_leading_article(text):
    return re.sub(r"^(the|a|an)\s+", "", text.strip(), flags=re.IGNORECASE)


def strip_both_prefix(text):
    return re.sub(r"^both\s+", "", text.strip(), flags=re.IGNORECASE)


def pluralize_object_word(word):
    if word.endswith("y"):
        return word[:-1] + "ies"
    return word + "s"


def singularize_plural_object(text):
    text = strip_leading_article(text)
    normalized = normalize_phrase(text)
    for object_word in sorted(OBJECT_WORDS, key=len, reverse=True):
        plural = pluralize_object_word(object_word)
        if normalized == normalize_phrase(plural):
            return object_word
        suffix = " " + normalize_phrase(plural)
        if normalized.endswith(suffix):
            prefix = text[: -len(plural)].strip()
            return f"{prefix} {object_word}".strip()
    return text


def split_task_clauses(task):
    parts = re.split(r"\s+and\s+(?=put\b|place\b|open\b|close\b|turn\b|pick\b|push\b)", task)
    return [part.strip() for part in parts if part.strip()]


def split_both_objects(object_text):
    had_both_prefix = object_text.strip().lower().startswith("both ")
    object_text = strip_leading_article(strip_both_prefix(object_text))
    if " and " not in object_text.lower():
        if had_both_prefix:
            singular = singularize_plural_object(object_text)
            if normalize_phrase(singular) != normalize_phrase(object_text):
                return [f"left {singular}", f"right {singular}"]
        return [strip_leading_article(object_text)]

    parts = re.split(r"\s+and\s+", object_text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return [strip_leading_article(object_text)]
    return [strip_leading_article(part) for part in parts if strip_leading_article(part)]


def split_manipulation_clause(clause):
    clause = clause.strip()
    action_match = re.match(r"^(put|place|pick up|open|close|turn on|turn off|push)\s+(.+)$", clause)
    if not action_match:
        return [{"clause": clause, "object": None, "target": None}]

    action = action_match.group(1)
    rest = action_match.group(2)
    best_index = None
    best_prep = None
    for prep in PREPOSITIONS:
        index = rest.find(prep)
        if index != -1 and (best_index is None or index < best_index):
            best_index = index
            best_prep = prep

    if best_index is None:
        for trailing_prep in TRAILING_PREPOSITIONS:
            suffix = f" {trailing_prep}"
            if rest.lower().endswith(suffix):
                obj = strip_leading_article(rest[: -len(suffix)])
                return [
                    {
                        "clause": clause,
                        "action": action,
                        "object": strip_leading_article(strip_both_prefix(obj)),
                        "target": None,
                        "preposition": trailing_prep,
                        "implicit_target": True,
                    }
                ]
        return [{"clause": clause, "action": action, "object": strip_leading_article(strip_both_prefix(rest)), "target": None}]

    obj = strip_leading_article(rest[:best_index])
    target = strip_leading_article(rest[best_index + len(best_prep) :])
    objects = split_both_objects(obj) if obj.lower().startswith("both ") else [strip_leading_article(obj)]
    if action == "pick up":
        return [
            {
                "clause": f"{action} the {obj_part} {best_prep.strip()} the {target}",
                "source_clause": clause,
                "action": action,
                "object": obj_part,
                "target": None,
                "source_location": target,
                "source_preposition": best_prep.strip(),
                "ambiguous_relation": "pickup_source_or_target",
            }
            for obj_part in objects
        ]

    return [
        {
            "clause": f"{action} the {obj_part} {best_prep.strip()} the {target}",
            "source_clause": clause,
            "action": action,
            "object": obj_part,
            "target": target,
            "preposition": best_prep.strip(),
        }
        for obj_part in objects
    ]


def rebuild_clause_text(clause):
    action = clause.get("action")
    obj = clause.get("object")
    target = clause.get("target")
    preposition = clause.get("preposition")
    if action and obj and target and preposition:
        return f"{action} the {obj} {preposition} the {target}"
    if action and obj:
        return f"{action} the {obj}"
    return clause.get("clause", "")


def refresh_allowed_phrases(constraints, task):
    allowed_phrases = []
    for clause in constraints["clauses"]:
        for key in ("object", "target", "source_location"):
            phrase = clause.get(key)
            if phrase:
                allowed_phrases.append(phrase)

    # Keep exact multi-color object names intact, e.g. "yellow and white mug".
    allowed_phrases.extend(
        match.group(0)
        for match in re.finditer(
            rf"\b(?:{'|'.join(COLOR_WORDS)})(?:\s+and\s+(?:{'|'.join(COLOR_WORDS)}))*\s+(?:{'|'.join(re.escape(word) for word in OBJECT_WORDS)})\b",
            task,
            flags=re.IGNORECASE,
        )
    )

    seen = set()
    unique_phrases = []
    for phrase in allowed_phrases:
        normalized = normalize_phrase(phrase)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_phrases.append(strip_leading_article(phrase))
    constraints["allowed_phrases"] = unique_phrases
    return constraints


def build_task_constraints(task):
    clauses = [
        split_clause
        for clause in split_task_clauses(task)
        for split_clause in split_manipulation_clause(clause)
    ]
    last_referent = None
    last_manipulated_object = None
    source_location_by_object = {}
    for clause in clauses:
        if normalize_phrase(clause.get("object", "")) == "it":
            if clause.get("action") in CONTROL_ACTIONS:
                candidates = []
                for candidate in (last_manipulated_object, last_referent):
                    if candidate and normalize_phrase(candidate) not in {
                        normalize_phrase(item) for item in candidates
                    }:
                        candidates.append(candidate)
                if candidates:
                    clause["pronoun_candidates"] = candidates
                    clause["ambiguous_pronoun"] = "control_object_it"
                referent = last_referent or last_manipulated_object
            else:
                referent = last_manipulated_object or last_referent
            if referent:
                clause["object"] = referent
                clause["resolved_pronoun"] = "it"
                clause["clause"] = rebuild_clause_text(clause)
        if normalize_phrase(clause.get("target", "")) == "it":
            if last_referent:
                clause["target"] = last_referent
                clause["resolved_pronoun"] = "it"
                clause["clause"] = rebuild_clause_text(clause)
        if (
            clause.get("implicit_target")
            and clause.get("action") in {"put", "place"}
            and not clause.get("target")
            and last_referent
        ):
            clause["target"] = last_referent
            clause["resolved_implicit_target"] = last_referent
            clause["clause"] = rebuild_clause_text(clause)

        object_key = normalize_phrase(clause.get("object", ""))
        if (
            clause.get("action") in {"put", "place"}
            and object_key in source_location_by_object
            and not clause.get("source_location")
        ):
            clause["source_location"] = source_location_by_object[object_key]["source_location"]
            clause["source_preposition"] = source_location_by_object[object_key]["source_preposition"]

        if clause.get("source_location") and object_key:
            source_location_by_object[object_key] = {
                "source_location": clause["source_location"],
                "source_preposition": clause.get("source_preposition") or "at",
            }

        if clause.get("object") and clause.get("action") not in CONTROL_ACTIONS:
            last_manipulated_object = clause["object"]

        if clause.get("target"):
            last_referent = clause["target"]
        elif clause.get("source_location"):
            last_referent = clause["source_location"]
        elif clause.get("object"):
            last_referent = clause["object"]

    return refresh_allowed_phrases({"clauses": clauses}, task)


def object_with_source_location(clause):
    obj = clause["object"]
    source_location = clause.get("source_location")
    if source_location:
        source_preposition = clause.get("source_preposition") or "at"
        return f"{obj} {source_preposition} the {source_location}"
    return obj


def build_recommended_outline(constraints, target_subtasks):
    pick_place_clauses, close_open_clauses = task_complexity_counts(constraints)
    if not pick_place_clauses and not close_open_clauses:
        return []

    outline = []
    if len(pick_place_clauses) == 1 and len(close_open_clauses) == 1 and target_subtasks == 3:
        put_clause = pick_place_clauses[0]
        control_clause = close_open_clauses[0]
        obj = put_clause["object"]
        obj_at_source = object_with_source_location(put_clause)
        target = put_clause["target"]
        preposition = put_clause.get("preposition") or "near"
        control_obj = control_clause["object"]
        action = control_clause["action"]
        return [
            f"{action} the {control_obj}",
            f"approach and grasp the {obj_at_source}",
            f"move and place the {obj} {preposition} the {target}",
        ]

    if len(pick_place_clauses) == 1 and len(close_open_clauses) == 1 and target_subtasks == 4:
        put_clause = pick_place_clauses[0]
        close_clause = close_open_clauses[0]
        obj = put_clause["object"]
        obj_at_source = object_with_source_location(put_clause)
        target = put_clause["target"]
        preposition = put_clause.get("preposition") or "near"
        close_obj = close_clause["object"]
        action = close_clause["action"]
        return [
            f"approach and grasp the {obj_at_source}",
            f"move and place the {obj} {preposition} the {target}",
            f"move the gripper to the {close_obj}",
            f"{action} the {close_obj}",
        ]

    if len(pick_place_clauses) == 1 and len(close_open_clauses) == 1 and target_subtasks == 5:
        put_clause = pick_place_clauses[0]
        close_clause = close_open_clauses[0]
        obj = put_clause["object"]
        obj_at_source = object_with_source_location(put_clause)
        target = put_clause["target"]
        preposition = put_clause.get("preposition") or "near"
        close_obj = close_clause["object"]
        action = close_clause["action"]
        return [
            f"approach the {obj_at_source}",
            f"grasp the {obj_at_source}",
            f"move and place the {obj} {preposition} the {target}",
            f"move the gripper to the {close_obj} handle or closing area",
            f"{action} the {close_obj}",
        ]

    if target_subtasks == len(pick_place_clauses) * 2:
        for clause in pick_place_clauses:
            obj = clause["object"]
            obj_at_source = object_with_source_location(clause)
            target = clause["target"]
            preposition = clause.get("preposition") or "near"
            outline.append(f"approach and grasp the {obj_at_source}")
            outline.append(f"move and place the {obj} {preposition} the {target}")
        return outline

    if target_subtasks == len(pick_place_clauses) * 2 + len(close_open_clauses):
        for clause in constraints["clauses"]:
            if clause in close_open_clauses:
                outline.append(f"{clause['action']} the {clause['object']}")
            elif clause in pick_place_clauses:
                obj = clause["object"]
                obj_at_source = object_with_source_location(clause)
                target = clause["target"]
                preposition = clause.get("preposition") or "near"
                outline.append(f"approach and grasp the {obj_at_source}")
                outline.append(f"move and place the {obj} {preposition} the {target}")
        return outline

    if target_subtasks == len(pick_place_clauses) * 4:
        for clause in pick_place_clauses:
            obj = clause["object"]
            obj_at_source = object_with_source_location(clause)
            target = clause["target"]
            preposition = clause.get("preposition") or "near"
            outline.extend(
                [
                    f"approach the {obj_at_source}",
                    f"grasp the {obj_at_source}",
                    f"move the {obj} above/near the {target}",
                    f"place or release the {obj} {preposition} the {target}",
                ]
            )
        return outline

    return []


def processor_pad_token_id(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    return pad_token_id


def ambiguous_pickup_clauses(constraints):
    return [
        clause
        for clause in constraints["clauses"]
        if clause.get("ambiguous_relation") == "pickup_source_or_target"
        and clause.get("object")
        and clause.get("source_location")
    ]


def ambiguous_pronoun_clauses(constraints):
    return [
        clause
        for clause in constraints["clauses"]
        if clause.get("ambiguous_pronoun") == "control_object_it"
        and clause.get("pronoun_candidates")
    ]


def merge_language_disambiguation(constraints, key, value):
    language_disambiguation = constraints.setdefault("language_disambiguation", {})
    language_disambiguation[key] = value
    if "status" not in language_disambiguation:
        language_disambiguation["status"] = "vlm_resolved"
    elif language_disambiguation["status"] != "fallback_source_location":
        language_disambiguation["status"] = "vlm_resolved"
    return constraints


def resolve_ambiguous_pronouns_with_vlm(
    model,
    processor,
    task,
    constraints,
    initial_image,
    enable_thinking=False,
):
    ambiguous_clauses = ambiguous_pronoun_clauses(constraints)
    if not ambiguous_clauses:
        return constraints

    candidate_lines = "\n".join(
        "- source_clause: {source_clause}; action: {action}; candidates: {candidates}".format(
            source_clause=clause.get("source_clause") or clause.get("clause"),
            action=clause.get("action"),
            candidates=", ".join(clause.get("pronoun_candidates", [])),
        )
        for clause in ambiguous_clauses
    )
    prompt = f"""You are resolving pronouns in a LIBERO robot manipulation instruction.

Some clauses contain a control action such as "open it" or "close it". The pronoun "it" may refer to the manipulated object, or to the container/target location that the object was placed into/on.

Use the full task wording and the initial top-view image if helpful. Return JSON only.

Task: {task}
Ambiguous pronoun candidates:
{candidate_lines}

Output schema:
{{
  "resolutions": [
    {{
      "source_clause": "exact source clause",
      "referent": "one exact candidate phrase"
    }}
  ]
}}

For patterns like "put X in/inside Y and close it", "it" usually refers to Y, because containers/appliances/drawers are things that can be closed. Do not choose X unless the instruction clearly says the object itself is opened or closed.
"""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a strict JSON generator. Output only valid JSON."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": initial_image},
            ],
        },
    ]

    import torch

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=processor_pad_token_id(processor),
        )

    input_token_len = inputs["input_ids"].shape[-1]
    text = processor.batch_decode(
        outputs.sequences[:, input_token_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    try:
        raw_resolution = extract_json_object(text)
    except (ValueError, json.JSONDecodeError):
        raw_resolution = {"resolutions": []}

    resolution_by_clause = {
        str(item.get("source_clause", "")).strip(): str(item.get("referent", "")).strip()
        for item in raw_resolution.get("resolutions", [])
        if isinstance(item, dict)
    }

    applied = []
    for clause in ambiguous_clauses:
        source_clause = clause.get("source_clause") or clause.get("clause")
        candidates = clause.get("pronoun_candidates", [])
        normalized_candidates = {normalize_phrase(candidate): candidate for candidate in candidates}
        referent = normalized_candidates.get(normalize_phrase(resolution_by_clause.get(source_clause)))
        if referent is None:
            referent = candidates[-1]

        clause["object"] = referent
        clause["vlm_pronoun_referent"] = referent
        clause["clause"] = rebuild_clause_text(clause)
        applied.append(
            {
                "source_clause": source_clause,
                "action": clause.get("action"),
                "candidates": candidates,
                "referent": referent,
            }
        )

    return merge_language_disambiguation(
        refresh_allowed_phrases(constraints, task),
        "pronoun_resolutions",
        {
            "resolutions": applied,
            "raw_model_response": text,
        },
    )


def resolve_ambiguous_pickup_relations_with_vlm(
    model,
    processor,
    task,
    constraints,
    initial_image,
    enable_thinking=False,
):
    ambiguous_clauses = ambiguous_pickup_clauses(constraints)
    if not ambiguous_clauses:
        return constraints

    candidate_lines = "\n".join(
        "- source_clause: {source_clause}; object: {object}; related phrase after preposition: {source_location}".format(
            **clause
        )
        for clause in ambiguous_clauses
    )
    prompt = f"""You are disambiguating a LIBERO robot instruction using the initial top-view image.

Some phrases have the form "pick up X on/inside/at Y". In these phrases, Y can mean either:
- source_location: Y only describes where X starts in the initial scene. The robot manipulates X, not Y.
- placement_target: Y is a semantic action target for X.

Use the initial image and the full task wording. Return JSON only.

Task: {task}
Ambiguous candidates:
{candidate_lines}

Output schema:
{{
  "resolutions": [
    {{
      "source_clause": "exact source clause",
      "relation_type": "source_location"
    }}
  ]
}}

Choose relation_type as "source_location" when X is visibly located on/inside/at Y at the start, especially when the task later says to place it somewhere else. Choose "placement_target" only if Y is the actual goal location for placing X.
"""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a strict JSON generator. Output only valid JSON."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": initial_image},
            ],
        },
    ]

    import torch

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=processor_pad_token_id(processor),
        )

    input_token_len = inputs["input_ids"].shape[-1]
    text = processor.batch_decode(
        outputs.sequences[:, input_token_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    try:
        raw_resolution = extract_json_object(text)
    except (ValueError, json.JSONDecodeError):
        merge_language_disambiguation(
            constraints,
            "pickup_relation_resolutions",
            {"status": "fallback_source_location", "raw_model_response": text},
        )
        return constraints

    resolution_by_clause = {
        str(item.get("source_clause", "")).strip(): str(item.get("relation_type", "")).strip()
        for item in raw_resolution.get("resolutions", [])
        if isinstance(item, dict)
    }

    applied = []
    for clause in ambiguous_clauses:
        relation_type = resolution_by_clause.get(clause.get("source_clause"), "source_location")
        if relation_type == "placement_target":
            clause["target"] = clause.pop("source_location")
            clause["preposition"] = clause.pop("source_preposition", "on")
            clause["clause"] = rebuild_clause_text(clause)
        clause["vlm_relation_type"] = relation_type
        applied.append(
            {
                "source_clause": clause.get("source_clause"),
                "object": clause.get("object"),
                "source_location": clause.get("source_location"),
                "target": clause.get("target"),
                "relation_type": relation_type,
            }
        )

    return merge_language_disambiguation(
        refresh_allowed_phrases(constraints, task),
        "pickup_relation_resolutions",
        {
            "resolutions": applied,
            "raw_model_response": text,
        },
    )


def resolve_final_close_start_with_vlm(
    model,
    processor,
    task,
    constraints,
    action_boundary_prior,
    candidate_keyframes,
    enable_thinking=False,
):
    if not action_boundary_prior or not candidate_keyframes:
        return action_boundary_prior

    close_open_clauses = [
        clause
        for clause in constraints["clauses"]
        if clause.get("action") in {"close", "open"} and clause.get("object")
    ]
    if len(close_open_clauses) != 1 or close_open_clauses[0].get("action") != "close":
        return action_boundary_prior
    if len(action_boundary_prior.get("boundary_frames", [])) < 2:
        return action_boundary_prior

    close_object = close_open_clauses[0]["object"]
    candidate_frames = [int(frame) for frame, _ in candidate_keyframes]
    prompt = f"""You are refining the boundary between two robot sub-tasks in a LIBERO episode.

Task: {task}
Close-action object: {close_object}

The previous sub-task is moving the gripper to the {close_object} handle or closing area.
The final sub-task starts when the gripper first physically engages the handle/door/closing area and begins the close motion.

Look at the provided top-view candidate frames in chronological order. Choose the earliest frame where the behavior has switched from "moving toward the handle" to "executing the close action".

For a drawer, cabinet, microwave, or door:
- The final close sub-task begins at the first contact/initial-force frame, not at the frame where the object is already mostly closed.
- Do not choose a frame merely because the gripper is still travelling freely toward the handle.
- Do choose the earliest frame where the gripper has reached the handle/door/closing contact area and the next motion is closing.
- If a candidate shows the drawer/cabinet/microwave already half closed or more, that candidate is too late. Choose an earlier candidate just before that visible large closure.
- If you are uncertain between two nearby candidate frames, choose the earlier one.

You must choose exactly one frame from this list:
{candidate_frames}

Return JSON only:
{{
  "start_frame": {candidate_frames[0]}
}}
"""
    content = [{"type": "text", "text": prompt}]
    for frame, row in candidate_keyframes:
        content.append({"type": "text", "text": f"Candidate frame {frame}:"})
        content.append({"type": "image", "image": row["_image"]})

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a strict JSON generator. Output only valid JSON."}],
        },
        {"role": "user", "content": content},
    ]

    import torch

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=processor_pad_token_id(processor),
        )

    input_token_len = inputs["input_ids"].shape[-1]
    text = processor.batch_decode(
        outputs.sequences[:, input_token_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    try:
        raw = extract_json_object(text)
        start_frame = int(raw["start_frame"])
    except (ValueError, json.JSONDecodeError, KeyError, TypeError):
        start_frame = action_boundary_prior["boundary_frames"][-1]

    raw_vlm_start_frame = min(candidate_frames, key=lambda frame: abs(frame - start_frame))
    nearest_frame = raw_vlm_start_frame
    close_object_text = close_object.lower()
    late_cap = None
    if any(word in close_object_text for word in ("drawer", "cabinet", "microwave", "door")):
        source_span = None
        if action_boundary_prior.get("boundary_reasons"):
            source_span = action_boundary_prior["boundary_reasons"][-1].get("source_span")
        if source_span:
            span_start, span_end = [int(value) for value in source_span]
            late_cap = round(span_start + 0.45 * (span_end - span_start))
            if nearest_frame > late_cap:
                capped_candidates = [frame for frame in candidate_frames if frame <= late_cap]
                if capped_candidates:
                    nearest_frame = capped_candidates[-1]

    refined_prior = dict(action_boundary_prior)
    refined_prior["boundary_frames"] = list(action_boundary_prior["boundary_frames"])
    refined_prior["boundary_frames"][-1] = nearest_frame
    refined_prior["boundary_reasons"] = list(action_boundary_prior.get("boundary_reasons", []))
    if refined_prior["boundary_reasons"]:
        refined_prior["boundary_reasons"][-1] = {
            **refined_prior["boundary_reasons"][-1],
            "frame": nearest_frame,
            "reason": "vlm_refined_close_start",
            "candidate_frames": candidate_frames,
            "late_cap_frame": late_cap,
            "raw_vlm_start_frame": raw_vlm_start_frame,
            "raw_model_response": text,
        }

    sampled_labels = []
    for frame in [label["frame"] for label in action_boundary_prior.get("sampled_keyframe_labels", [])]:
        subtask_idx = 0
        for idx, boundary in enumerate(refined_prior["boundary_frames"]):
            if frame >= boundary:
                subtask_idx = idx
        sampled_labels.append({"frame": frame, "subtask_id": f"S{subtask_idx + 1}"})
    refined_prior["sampled_keyframe_labels"] = sampled_labels
    return refined_prior


def find_disallowed_phrases(annotation, allowed_phrases):
    allowed = {normalize_phrase(phrase) for phrase in allowed_phrases}
    object_pattern = "|".join(re.escape(word) for word in sorted(OBJECT_WORDS, key=len, reverse=True))
    color_pattern = "|".join(COLOR_WORDS)
    phrase_pattern = re.compile(
        rf"\b(?:{color_pattern})(?:\s+and\s+(?:{color_pattern}))*\s+(?:{object_pattern})\b",
        flags=re.IGNORECASE,
    )

    disallowed = set()
    for subtask in annotation.get("subtasks", []):
        text = str(subtask.get("text", ""))
        for match in phrase_pattern.finditer(text):
            phrase = normalize_phrase(match.group(0))
            if phrase not in allowed:
                disallowed.add(match.group(0).lower())
    return sorted(disallowed)


def find_mentioned_phrases(text, phrases):
    mentions = []
    occupied = []
    for phrase in sorted(phrases, key=len, reverse=True):
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", text):
            span = match.span()
            if any(not (span[1] <= used[0] or span[0] >= used[1]) for used in occupied):
                continue
            occupied.append(span)
            mentions.append(phrase)
    return mentions


def build_validation_feedback(annotation, constraints, target_subtasks, min_subtask_frames=1):
    feedback = []
    disallowed_phrases = find_disallowed_phrases(annotation, constraints["allowed_phrases"])
    if disallowed_phrases:
        feedback.append(
            "The following generated object phrases are not allowed by the task: "
            + ", ".join(disallowed_phrases)
            + ". Allowed exact phrases are: "
            + ", ".join(constraints["allowed_phrases"])
            + ". Preserve compound color phrases exactly, such as 'yellow and white mug'."
        )

    subtasks = annotation.get("subtasks", [])
    subtask_text_blob = " ".join(normalize_phrase(subtask.get("text", "")) for subtask in subtasks)
    for clause in constraints["clauses"]:
        action = clause.get("action")
        obj = normalize_phrase(clause.get("object", ""))
        if action in CONTROL_ACTIONS and (
            not re.search(rf"\b{re.escape(action)}\b", subtask_text_blob)
            or (obj and not re.search(rf"\b{re.escape(obj)}\b", subtask_text_blob))
        ):
            feedback.append(
                f"The instruction includes '{action} the {clause.get('object')}', "
                "so one subtask must explicitly include that action and object."
            )

    manipulated_objects = {
        normalize_phrase(clause["object"])
        for clause in constraints["clauses"]
        if clause.get("object") and clause.get("action") not in CONTROL_ACTIONS
    }
    for subtask in subtasks:
        text = normalize_phrase(subtask.get("text", ""))
        mentioned_objects = find_mentioned_phrases(text, manipulated_objects)
        if " both " in f" {text} " or len(mentioned_objects) > 1:
            feedback.append(
                "Each subtask must manipulate exactly one task object. "
                "Do not combine multiple objects with 'both' or 'and' in one subtask."
            )
            break

    if len(subtasks) != target_subtasks:
        feedback.append(
            f"The JSON must contain exactly {target_subtasks} subtasks, but it contained "
            f"{len(subtasks)}. Do not create one subtask per keyframe."
        )

    if min_subtask_frames > 1:
        short_subtasks = []
        for subtask in subtasks:
            try:
                duration = int(subtask["end_frame"]) - int(subtask["start_frame"]) + 1
            except (KeyError, TypeError, ValueError):
                continue
            if duration < min_subtask_frames:
                short_subtasks.append(
                    f"{subtask.get('subtask_id', '?')}({duration} frames: "
                    f"{subtask.get('start_frame')}-{subtask.get('end_frame')})"
                )
        if short_subtasks:
            feedback.append(
                f"Sub-task segments shorter than {min_subtask_frames} frames are suspicious "
                "and usually indicate a boundary mismatch. Reassign nearby keyframes so each "
                "sub-task covers a real continuous manipulation phase. Too-short segments: "
                + ", ".join(short_subtasks)
                + "."
            )

    sampled_keyframe_labels = annotation.get("sampled_keyframe_labels", [])
    if not sampled_keyframe_labels:
        feedback.append(
            "The JSON must include keyframe_labels assigning every sampled keyframe to a subtask."
        )
    else:
        id_to_order = {subtask["subtask_id"]: idx for idx, subtask in enumerate(subtasks)}
        orders = [
            id_to_order[label["subtask_id"]]
            for label in sampled_keyframe_labels
            if label.get("subtask_id") in id_to_order
        ]
        if any(curr < prev for prev, curr in zip(orders, orders[1:])):
            feedback.append(
                "Sampled keyframe labels must be chronological and non-decreasing by subtask_id."
            )
        labeled_ids = {label.get("subtask_id") for label in sampled_keyframe_labels}
        missing_ids = [
            subtask["subtask_id"]
            for subtask in subtasks
            if subtask["subtask_id"] not in labeled_ids
        ]
        if missing_ids:
            feedback.append(
                "Each subtask must be assigned to at least one sampled keyframe. "
                f"Missing labels for: {', '.join(missing_ids)}."
            )

    normalized_clause_texts = [normalize_phrase(clause["clause"]) for clause in constraints["clauses"]]
    normalized_subtask_texts = [normalize_phrase(subtask.get("text", "")) for subtask in subtasks]
    for clause_text in normalized_clause_texts:
        exact_repetitions = sum(1 for text in normalized_subtask_texts if text == clause_text)
        if exact_repetitions > 1:
            feedback.append(
                "Do not repeat the full task clause as multiple separate subtasks. "
                "Use semantic phases such as approach, grasp, move/transport, and place/release."
            )
            break

    return feedback


def build_messages(task, keyframes, target_subtasks, constraints, action_boundary_prior=None, retry_feedback=None):
    allowed_lines = "\n".join(f"- {phrase}" for phrase in constraints["allowed_phrases"])
    clause_lines = "\n".join(
        f"- {clause['clause']}" for clause in constraints["clauses"]
    )
    outline = build_recommended_outline(constraints, target_subtasks)
    outline_lines = "\n".join(f"S{idx}: {text}" for idx, text in enumerate(outline, start=1))
    outline_text = ""
    if outline_lines:
        outline_text = f"""
Recommended semantic outline. Use these exact objects/targets and keep this order; assign sampled keyframes to these subtask IDs:
{outline_lines}
"""
    action_prior_text = ""
    if action_boundary_prior:
        span_lines = "\n".join(
            "- frames {start_frame}-{end_frame}: gripper_command={gripper_command}".format(**span)
            for span in action_boundary_prior["spans"]
        )
        label_lines = "\n".join(
            f"- frame {label['frame']}: {label['subtask_id']}"
            for label in action_boundary_prior["sampled_keyframe_labels"]
        )
        action_prior_text = f"""
Robot action boundary prior from gripper-command transitions. Use this as a strong prior when assigning keyframe_labels:
{span_lines}
Recommended keyframe_labels from action prior:
{label_lines}
"""
    retry_text = ""
    if retry_feedback:
        retry_text = f"""

Previous response failed validation:
{retry_feedback}
Regenerate the JSON and fix these issues. Do not repeat any disallowed phrase. Start with {{ and end with }}.
"""

    prompt = f"""You are annotating a robot manipulation episode from the LIBERO dataset.

The episode has one overall task instruction and a sequence of top-view keyframes. The keyframes are sampled from the episode timeline. Ignore any missing hand-camera view; use only the provided top-view images.

Your job:
1. Split the full episode into exactly {target_subtasks} chronological sub-tasks.
2. For each sub-task, write a short English manipulation description.
3. Assign every sampled keyframe to exactly one sub-task using keyframe_labels.
4. The keyframe_labels must be chronological and non-decreasing: once the episode moves from S1 to S2, it must not go back to S1.
5. keyframe_labels must contain every sampled frame exactly once: {[frame for frame, _ in keyframes]}.
6. The code will derive start_frame/end_frame from keyframe_labels; therefore choose keyframe_labels carefully from visual evidence.
7. Use the task instruction as the source of truth for object names and target locations.
8. Do not introduce objects, colors, containers, or target locations that are not named in the task instruction.
9. Treat every phrase in "Allowed exact object/location phrases" as atomic. For example, "yellow and white mug" is one object, not "yellow mug" plus "white mug".
10. Every object or target phrase in subtask text must be copied from the allowed exact phrases below, or be a generic robot term such as gripper/end-effector.
11. Do not create one sub-task per sampled keyframe. Merge adjacent keyframes that show the same semantic phase.
12. For a simple pick-and-place task, use phases like approach the object, grasp the object, move the object above the target, and release/place the object.
13. Avoid vague repeated text such as "move the gripper to the target object" after the object has already been grasped.
14. For multi-clause tasks, complete the semantic phases for one task clause before moving to the next task clause. Do not alternate the same clauses repeatedly across keyframes.
15. If the instruction says "both A and B", annotate A and B as separate manipulated objects, not as one combined object.
16. If the instruction includes an extra open/close/turn-on/turn-off action, keep that control action as its own sub-task instead of merging it with a pick-and-place sub-task.
17. Do not make a brief gripper release its own sub-task when a later open/close action exists. Include release in the object placement sub-task, then use the next sub-task for moving the gripper to the handle or closing area.
18. If the instruction first says to turn on or turn off an appliance and then place an object on it, annotate the appliance control action before the object approach/grasp/place phases.
19. Use only visible evidence from the top-view images plus the task instruction. If the exact boundary is uncertain, choose the nearest sampled keyframe boundary conservatively.
20. Return valid JSON only. Do not wrap it in Markdown.
21. Do not write analysis, reasoning, commentary, or explanations. Your entire response must start with {{ and end with }}.

Output schema:
{{
  "subtasks": [
    {{
      "subtask_id": "S1",
      "text": "move the gripper to the target object"
    }}
  ],
  "keyframe_labels": [
    {{"frame": {keyframes[0][0]}, "subtask_id": "S1"}}
  ]
}}

Task instruction: {task}
Task clauses:
{clause_lines}
Allowed exact object/location phrases:
{allowed_lines}
{outline_text}
{action_prior_text}
Sampled keyframe frame indices in image order: {[frame for frame, _ in keyframes]}
{retry_text}
"""

    content = [{"type": "text", "text": prompt}]
    for frame, row in keyframes:
        content.append({"type": "text", "text": f"Keyframe at original frame {frame}:"})
        content.append({"type": "image", "image": row["_image"]})
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a strict JSON generator. Output only valid JSON. Do not explain your reasoning.",
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def generate_annotation(
    model,
    processor,
    task,
    keyframes,
    target_subtasks,
    constraints,
    max_new_tokens,
    enable_thinking,
    action_boundary_prior=None,
    retry_feedback=None,
):
    import torch

    messages = build_messages(
        task,
        keyframes,
        target_subtasks,
        constraints,
        action_boundary_prior=action_boundary_prior,
        retry_feedback=retry_feedback,
    )

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=processor_pad_token_id(processor),
        )

    input_token_len = inputs["input_ids"].shape[-1]
    answer_token_ids = outputs.sequences[:, input_token_len:]
    return processor.batch_decode(
        answer_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def extract_json_object(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return json.loads(text[start : end + 1])

    raise ValueError(f"Model response did not contain a JSON object:\n{text}")


def normalize_sampled_keyframe_labels(raw_annotation, subtask_ids, keyframe_frames):
    raw_labels = raw_annotation.get("keyframe_labels") or raw_annotation.get("sampled_keyframe_labels")
    if not isinstance(raw_labels, list):
        return []

    keyframe_set = set(keyframe_frames)
    label_by_frame = {}
    for label in raw_labels:
        if not isinstance(label, dict):
            continue
        try:
            frame = int(label["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        subtask_id = str(label.get("subtask_id", "")).strip()
        if frame in keyframe_set and subtask_id in subtask_ids:
            label_by_frame[frame] = subtask_id

    if not label_by_frame:
        return []

    raw_subtasks = raw_annotation.get("subtasks") or []
    raw_ranges = []
    for idx, subtask in enumerate(raw_subtasks, start=1):
        subtask_id = f"S{idx}"
        if "start_frame" not in subtask or "end_frame" not in subtask:
            continue
        try:
            raw_ranges.append(
                (
                    int(subtask["start_frame"]),
                    int(subtask["end_frame"]),
                    subtask_id,
                )
            )
        except (TypeError, ValueError):
            continue

    sampled_labels = []
    for frame in keyframe_frames:
        subtask_id = label_by_frame.get(frame)
        if subtask_id is None:
            for start, end, range_subtask_id in raw_ranges:
                if start <= frame <= end:
                    subtask_id = range_subtask_id
                    break
        if subtask_id is None:
            return []
        sampled_labels.append({"frame": frame, "subtask_id": subtask_id})

    return sampled_labels


def apply_keyframe_labels_to_subtasks(
    normalized_subtasks,
    sampled_keyframe_labels,
    first_frame,
    last_frame,
    boundary_frames=None,
):
    if not sampled_keyframe_labels and not boundary_frames:
        return normalized_subtasks

    first_frame_by_subtask = {
        f"S{idx + 1}": int(frame)
        for idx, frame in enumerate(boundary_frames or [])
    }
    for label in sampled_keyframe_labels:
        first_frame_by_subtask.setdefault(label["subtask_id"], int(label["frame"]))

    previous_end = first_frame - 1
    for idx, subtask in enumerate(normalized_subtasks):
        subtask_id = subtask["subtask_id"]
        if idx == 0:
            start = first_frame
        else:
            start = first_frame_by_subtask.get(subtask_id, previous_end + 1)
            start = max(previous_end + 1, min(start, last_frame))

        next_start = None
        for next_subtask in normalized_subtasks[idx + 1 :]:
            next_start = first_frame_by_subtask.get(next_subtask["subtask_id"])
            if next_start is not None:
                break

        end = last_frame if next_start is None else max(start, min(next_start - 1, last_frame))
        subtask["start_frame"] = start
        subtask["end_frame"] = end
        previous_end = end

    return [
        subtask
        for subtask in normalized_subtasks
        if subtask["start_frame"] <= subtask["end_frame"]
    ]


def normalize_annotation(
    raw_annotation,
    episode_id,
    task,
    first_frame,
    last_frame,
    keyframe_frames=None,
    action_boundary_prior=None,
):
    subtasks = raw_annotation.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        raise ValueError("Model JSON must contain a non-empty 'subtasks' list.")

    normalized_subtasks = []
    next_start = first_frame
    for idx, subtask in enumerate(subtasks, start=1):
        start = int(subtask.get("start_frame", next_start))
        end = int(subtask.get("end_frame", start))
        start = max(first_frame, min(start, last_frame))
        end = max(start, min(end, last_frame))
        if normalized_subtasks:
            start = normalized_subtasks[-1]["end_frame"] + 1
            end = max(start, end)

        normalized_subtasks.append(
            {
                "subtask_id": f"S{idx}",
                "text": str(subtask.get("text", "")).strip() or f"sub-task {idx}",
                "start_frame": start,
                "end_frame": end,
            }
        )
        next_start = end + 1

    normalized_subtasks[0]["start_frame"] = first_frame
    normalized_subtasks[-1]["end_frame"] = last_frame
    for idx in range(1, len(normalized_subtasks)):
        normalized_subtasks[idx]["start_frame"] = normalized_subtasks[idx - 1]["end_frame"] + 1
    for idx in range(len(normalized_subtasks) - 2, -1, -1):
        if normalized_subtasks[idx]["end_frame"] >= normalized_subtasks[idx + 1]["start_frame"]:
            normalized_subtasks[idx]["end_frame"] = normalized_subtasks[idx + 1]["start_frame"] - 1

    subtask_ids = {subtask["subtask_id"] for subtask in normalized_subtasks}
    sampled_keyframe_labels = normalize_sampled_keyframe_labels(
        raw_annotation,
        subtask_ids,
        keyframe_frames or [],
    )
    normalized_subtasks = apply_keyframe_labels_to_subtasks(
        normalized_subtasks,
        sampled_keyframe_labels,
        first_frame,
        last_frame,
        boundary_frames=(action_boundary_prior or {}).get("boundary_frames"),
    )

    normalized_subtasks = [
        subtask
        for subtask in normalized_subtasks
        if subtask["start_frame"] <= subtask["end_frame"]
    ]

    frame_labels = []
    for subtask in normalized_subtasks:
        for frame in range(subtask["start_frame"], subtask["end_frame"] + 1):
            frame_labels.append({"frame": frame, "subtask_id": subtask["subtask_id"]})

    return {
        "episode_id": episode_id,
        "task": task,
        "subtasks": normalized_subtasks,
        "sampled_keyframe_labels": sampled_keyframe_labels,
        "action_boundary_prior": action_boundary_prior,
        "frame_labels": frame_labels,
    }


def save_outputs(output_dir, annotation, raw_response, keyframes, save_keyframes, raw_attempts=None):
    episode_id = annotation["episode_id"]
    episode_dir = output_dir / f"episode_{episode_id:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = episode_dir / "annotation.json"
    raw_path = episode_dir / "raw_model_response.txt"
    keyframes_path = episode_dir / "keyframes.json"

    for stale_path in episode_dir.glob("raw_model_response_attempt_*.txt"):
        stale_path.unlink()

    annotation_path.write_text(
        json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    raw_path.write_text(raw_response + "\n", encoding="utf-8")
    if raw_attempts:
        for attempt_index, attempt_response in enumerate(raw_attempts, start=1):
            (episode_dir / f"raw_model_response_attempt_{attempt_index}.txt").write_text(
                attempt_response + "\n",
                encoding="utf-8",
            )
    keyframes_path.write_text(
        json.dumps([{"frame": frame} for frame, _ in keyframes], indent=2) + "\n",
        encoding="utf-8",
    )

    if save_keyframes:
        keyframe_dir = episode_dir / "keyframes"
        keyframe_dir.mkdir(exist_ok=True)
        for frame, row in keyframes:
            row["_image"].save(keyframe_dir / f"frame_{frame:06d}.png")

    return annotation_path


def save_failed_outputs(output_dir, episode_id, raw_attempts):
    episode_dir = output_dir / f"episode_{episode_id:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in episode_dir.glob("raw_model_response_attempt_*.txt"):
        stale_path.unlink()
    for attempt_index, attempt_response in enumerate(raw_attempts, start=1):
        (episode_dir / f"raw_model_response_attempt_{attempt_index}.txt").write_text(
            attempt_response + "\n",
            encoding="utf-8",
        )
    return episode_dir


def main():
    args = parse_args()
    if not args.local_data_dir.exists():
        raise FileNotFoundError(f"Local data directory does not exist: {args.local_data_dir}")
    if args.num_keyframes < 2:
        raise ValueError("--num-keyframes must be at least 2")

    task_map = load_task_map(args.local_data_dir)
    episode_df = load_episode(args.local_data_dir, args.episode_id)
    task_index = int(scalar_from_value(episode_df.iloc[0]["task_index"]))
    task = task_map[task_index]
    constraints = build_task_constraints(task)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if ambiguous_pickup_clauses(constraints) or ambiguous_pronoun_clauses(constraints):
        initial_image = image_from_value(episode_df.iloc[0][TOP_VIEW_KEY], args.local_data_dir)
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

    if args.max_vlm_keyframes < 2:
        raise ValueError("--max-vlm-keyframes must be at least 2")

    preliminary_action_boundary_prior = None
    if not args.disable_action_boundary_prior:
        preliminary_action_boundary_prior = build_action_boundary_prior(
            episode_df=episode_df,
            constraints=constraints,
            target_subtasks=target_subtasks,
            keyframe_frames=[],
            min_gripper_span_frames=args.min_gripper_span_frames,
        )

    effective_keyframes = min(args.num_keyframes, args.max_vlm_keyframes)
    keyframes = choose_action_aware_keyframe_rows(
        episode_df,
        effective_keyframes,
        boundary_frames=(preliminary_action_boundary_prior or {}).get("boundary_frames"),
    )
    prepared_keyframes = []
    for frame, row in keyframes:
        row = row.copy()
        row["_image"] = image_from_value(row[TOP_VIEW_KEY], args.local_data_dir)
        prepared_keyframes.append((frame, row))
    action_boundary_prior = None
    if not args.disable_action_boundary_prior:
        action_boundary_prior = build_action_boundary_prior(
            episode_df=episode_df,
            constraints=constraints,
            target_subtasks=target_subtasks,
            keyframe_frames=[frame for frame, _ in prepared_keyframes],
            min_gripper_span_frames=args.min_gripper_span_frames,
        )

    raw_attempts = []
    retry_feedback = None
    annotation = None
    validation_feedback = []
    for attempt in range(args.max_retries + 1):
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
                "Do not analyze the images in prose. Output only one JSON object that starts with { and ends with }."
            )
            validation_feedback = [retry_feedback]
            continue

        annotation = normalize_annotation(
            raw_annotation=raw_annotation,
            episode_id=args.episode_id,
            task=task,
            first_frame=int(episode_df["frame_index"].min()),
            last_frame=int(episode_df["frame_index"].max()),
            keyframe_frames=[frame for frame, _ in prepared_keyframes],
            action_boundary_prior=action_boundary_prior,
        )
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
        failed_dir = save_failed_outputs(args.output_dir, args.episode_id, raw_attempts)
        raise RuntimeError(
            "Failed to produce a JSON annotation after retries. "
            f"Raw attempts were saved under {failed_dir}."
        )

    annotation_path = save_outputs(
        output_dir=args.output_dir,
        annotation=annotation,
        raw_response=raw_response,
        keyframes=prepared_keyframes,
        save_keyframes=args.save_keyframes,
        raw_attempts=raw_attempts,
    )

    print(f"Episode {args.episode_id}: {task}")
    print(f"Frames: {annotation['frame_labels'][0]['frame']}..{annotation['frame_labels'][-1]['frame']}")
    print(f"Subtasks: {len(annotation['subtasks'])}")
    print(f"Target subtasks: {target_subtasks}")
    print(f"Allowed phrases: {constraints['allowed_phrases']}")
    if action_boundary_prior:
        print(f"Action boundary prior: {action_boundary_prior['boundary_frames']}")
    if validation_feedback:
        print("Warning: validation issues remained after retries:")
        for issue in validation_feedback:
            print(f"- {issue}")
    print(f"Saved annotation: {annotation_path}")


if __name__ == "__main__":
    main()
