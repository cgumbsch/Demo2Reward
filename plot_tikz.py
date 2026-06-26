"""Generate TikZ-ready reward plots by running the critic on demo data.

Supports two modes:
  - Zero-shot:    uses the dataset's task description directly (no refined prompt).
  - Demo2Reward:  uses a refined instruction (passed via --instruction).

Examples
--------
# Zero-shot:
python plot_tikz.py --task metaworld_boxclose --data_root ./data

# Demo2Reward (with a refined instruction):
python plot_tikz.py --task metaworld_boxclose --data_root ./data \
    --instruction "In the final frame, determine if the robot ..."

# Real robot:
python plot_tikz.py --task real_lid --data_root ./data --real_robot \
    --instruction "The robot's task is to place the lid ..."
"""

import os
import argparse

from utils.prompts import build_prompts, build_critic_user_prompt, build_zero_shot_user_prompt
from utils.critic import eval_single, _parse_binary
from utils.plots import log_ab_plot_tikz
from vlms.vlm_utils import get_vlm_funcs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot critic predictions vs GT as TikZ + PNG",
    )

    # Data / output
    p.add_argument("--data_root", default="./data",
                   help="Root directory holding the demonstration datasets")
    p.add_argument("--output_dir", default="./plots",
                   help="Directory for the generated TikZ/PNG plots")

    # Model
    p.add_argument("--vlm", default="qwen3_8b", choices=["qwen3_8b", "qwen3_32b"])

    # Task / dataset
    p.add_argument("--task", default="metaworld_boxclose")
    p.add_argument("--past_len", default=4, type=int)
    p.add_argument("--num_demos", default=8, type=int,
                   help="Number of demo indices to iterate over")
    p.add_argument("--success_once_is_enough", action="store_true", default=True)

    # Prompt
    p.add_argument("--real_robot", action="store_true",
                   help="Use 'a robot' instead of 'a simulated robot' in prompts")
    p.add_argument("--instruction", default=None, type=str,
                   help="Refined instruction (Demo2Reward mode). "
                        "Omit for zero-shot evaluation.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading (one demo at a time, varying ref_i)
# ---------------------------------------------------------------------------

def load_dataset_for_demo(suite, task, ref_i, args, data_root, past_len):
    """Load a dataset configured around a single reference demo *ref_i*."""
    common = dict(
        data_root=data_root,
        task=task,
        ref_i=ref_i,
        num_demos=2,
        num_test=0,
        past_len=past_len,
        video_mode=args.success_once_is_enough,
        full_past=False,
        enhance_data_factor=0,
    )

    if suite == "metaworld":
        from datasets.metaworld import MetaworldDataset, get_task_instruction
        common["num_test"] = 1
        return MetaworldDataset(**common), get_task_instruction(task)

    if suite == "robomimic":
        from datasets.robomimic import RobomimicDataset, get_task_instruction
        return RobomimicDataset(**common), get_task_instruction(task)

    if suite == "real":
        from datasets.real_rlinf import RealRLInfDataset, get_task_instruction
        return RealRLInfDataset(**common), get_task_instruction(task)

    raise ValueError(f"Unsupported suite for tikz plots: {suite}")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def eval_videos_with_prompt(model, processor, prompt_fn, critic_sys_prompt,
                            user_prompt, videos):
    """Score a list of videos using a pre-built user prompt."""
    rewards = []
    for video in videos:
        raw = eval_single(
            model, processor, prompt_fn, critic_sys_prompt,
            user_prompt, video, debug=False,
        )
        rewards.append(int(_parse_binary(raw)))
    return rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Prompts
    _, critic_sys_prompt, _ = build_prompts(args.real_robot)
    use_demo2reward = args.instruction is not None

    # Load model
    model, processor, prompt_vlm = get_vlm_funcs(name=args.vlm)
    model.eval()
    print("Starting plots")

    # Output directory
    mode_dir = "demo2reward_vlm_critic" if use_demo2reward else "vlm_critic"
    suite, task = args.task.split("_", 1)

    for i in range(args.num_demos):
        print(f"I= {i}")

        dataset, task_instruction = load_dataset_for_demo(
            suite, task, ref_i=i, args=args,
            data_root=args.data_root, past_len=args.past_len,
        )

        # Build the user prompt
        if use_demo2reward:
            user_prompt = build_critic_user_prompt(args.instruction)
        else:
            user_prompt = build_zero_shot_user_prompt(task_instruction)

        # Get example videos and GT from validation split
        videos, gt_successes = dataset.get_val_positive_example_video()

        print("Evaluating videos")
        predictions = eval_videos_with_prompt(
            model, processor, prompt_vlm, critic_sys_prompt, user_prompt, videos,
        )

        print("Saving plot")
        root_path = f"{args.output_dir}/{mode_dir}/{args.task}"
        os.makedirs(root_path, exist_ok=True)

        log_ab_plot_tikz(
            f"demo_{i}",
            a=gt_successes, a_label="GT success",
            b=predictions, b_label="Pred. success",
            out_dir=root_path,
        )


if __name__ == "__main__":
    main()
