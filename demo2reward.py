"""Demo2Reward: test-time adaptation of VLM reward instructions.

Iteratively refines a task-success instruction using a Meta-Critic / Critic
loop on a small set of expert demonstrations.
"""

import os
import argparse
from datetime import datetime
import random
import pickle

import numpy as np

from utils.logging import TensorBoardLogger
from utils.evaluate_prompts import compute_score
from utils.prompts import build_prompts, build_generic_prompt
from utils.critic import evaluate_prompt, refine_prompt, plot_examples
from vlms.vlm_utils import prompt_strategy, get_vlm_funcs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Demo2Reward: test-time adaptation of VLM reward instructions",
    )

    # Data
    p.add_argument("--data_root", default="./data",
                   help="Root directory holding the demonstration datasets")

    # Models
    p.add_argument("--vlm", default="qwen3_8b", choices=["qwen3_8b", "qwen3_32b"],
                   help="Critic VLM")
    p.add_argument("--meta_vlm", default=None,
                   choices=[None, "qwen3_8b", "qwen3_32b"],
                   help="Meta-Critic VLM (defaults to --vlm for single-model mode)")

    # Task / dataset
    p.add_argument("--task", default="metaworld_assembly")
    p.add_argument("--num_demos", default=3, type=int)
    p.add_argument("--dataset_factor", default=0, type=int)
    p.add_argument("--success_once_is_enough", action="store_true", default=True)

    # Optimisation
    p.add_argument("--objective", default="extreme_weighted_sum_rates")
    p.add_argument("--objective_threshold", default=2.0, type=float)
    p.add_argument("--num_repeats", default=5, type=int)
    p.add_argument("--start_index", default=0, type=int)
    p.add_argument("--max_iter", default=100, type=int)
    p.add_argument("--prompt_strategy", default="explore")
    p.add_argument("--past_len", default=4, type=int)
    p.add_argument("--test_every", default=20, type=int)

    # Decision mode
    p.add_argument("--decision_mode", default="deter",
                   choices=["deter", "stoch", "sample"])
    p.add_argument("--decision_threshold", default=0.8, type=float)
    p.add_argument("--decision_samples", default=10, type=int)

    # Prompt wording
    p.add_argument("--real_robot", action="store_true",
                   help="Use 'a robot' instead of 'a simulated robot' in prompts")

    # Logging
    p.add_argument("--output_dir", default="./logs",
                   help="Directory for TensorBoard logs and result files")
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--wandb", action="store_true",
                   help="Also log to Weights & Biases (requires wandb to be installed and authenticated)")
    p.add_argument("--wandb_project", default="demo2reward",
                   help="W&B project name (used when --wandb is set)")
    p.add_argument("--wandb_entity", default=None,
                   help="W&B entity/team name (used when --wandb is set)")

    args = p.parse_args()

    # Default meta_vlm → vlm  (single-model mode)
    if args.meta_vlm is None:
        args.meta_vlm = args.vlm

    return args


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(suite, task, args, data_root, past_len):
    """Return ``(dataset, task_instruction)``."""
    common = dict(
        data_root=data_root,
        task=task,
        ref_i=0,
        num_demos=args.num_demos,
        past_len=past_len,
        video_mode=args.success_once_is_enough,
        enhance_data_factor=args.dataset_factor,
    )

    if suite == "metaworld":
        from datasets.metaworld import MetaworldDataset, get_task_instruction
        dataset = MetaworldDataset(num_test=5, **common)
        return dataset, get_task_instruction(task)

    if suite == "robomimic":
        from datasets.robomimic import RobomimicDataset, get_task_instruction
        num_test = 10 if (task == "square" and args.num_demos <= 40) else 0
        dataset = RobomimicDataset(num_test=num_test, **common)
        return dataset, get_task_instruction(task)

    if suite == "real":
        from datasets.real_rlinf import RealRLInfDataset, get_task_instruction
        common["ref_i"] = 3
        dataset = RealRLInfDataset(num_test=5, **common)
        return dataset, get_task_instruction(task)

    raise ValueError(f"Unknown suite: {suite}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args):
    """Load critic (and optionally separate meta-critic) VLMs.

    Returns ``(model, processor, prompt_vlm,
               meta_model, meta_processor, meta_prompt_vlm)``.
    """
    use_separate = args.vlm != args.meta_vlm

    model, processor, prompt_vlm = get_vlm_funcs(name=args.vlm)

    if use_separate:
        meta_model, meta_processor, meta_prompt_vlm = get_vlm_funcs(name=args.meta_vlm)
        meta_model.eval()
    else:
        meta_model, meta_processor, meta_prompt_vlm = model, processor, prompt_vlm

    model.eval()

    # Override critic prompt function for stochastic decision modes
    if args.decision_mode != "deter":
        assert "qwen" in args.vlm, "Stochastic decision modes require a Qwen model"
        from vlms.qwen_utils import qwen_binary_cap_prob, qwen_binary_cap_samples
        n = args.decision_samples
        t = args.decision_threshold
        if args.decision_mode == "stoch":
            prompt_vlm = lambda model, processor, messages, prompt_kwargs, debug: \
                qwen_binary_cap_prob(model, processor, messages, prompt_kwargs, debug,
                                     threshold=t, num_samples=n)
        else:
            prompt_vlm = lambda model, processor, messages, prompt_kwargs, debug: \
                qwen_binary_cap_samples(model, processor, messages, prompt_kwargs, debug,
                                        threshold=t, num_samples=n)

    return model, processor, prompt_vlm, meta_model, meta_processor, meta_prompt_vlm


# ---------------------------------------------------------------------------
# Run-name construction
# ---------------------------------------------------------------------------

def build_run_name(args, use_separate_models):
    date_str = datetime.now().strftime("%Y%m%d")
    parts = [
        args.task,
        f"meta-{args.meta_vlm}_" if use_separate_models else "",
        f"{args.decision_mode}-{args.decision_samples}_" if args.decision_mode != "deter" else "",
        args.vlm,
        f"_{args.num_demos}demos_" if args.num_demos != 3 else "",
        f"_{args.num_repeats}-by-{args.max_iter}",
        f"-x{args.dataset_factor}_" if args.dataset_factor > 1 else "",
        f"_{args.objective}_success",
        f"_frames-{args.past_len}",
        f"_{args.prompt_strategy}",
        f"_{date_str}",
    ]
    return "".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Prompts
    meta_critic_prompt, critic_prompt, robot_desc = build_prompts(args.real_robot)
    mc_prompt_kwargs = prompt_strategy[args.prompt_strategy]

    # Convenience aliases
    objective = args.objective
    compute_objective = lambda s: compute_score(s, objective=objective,
                                                threshold=args.objective_threshold)

    # Load models
    (model, processor, prompt_vlm,
     meta_model, meta_processor, meta_prompt_vlm) = load_models(args)

    # Load dataset
    suite, task = args.task.split("_", 1)
    dataset, task_instruction = load_dataset(
        suite, task, args, args.data_root, args.past_len,
    )

    # Output directory
    use_separate_models = args.vlm != args.meta_vlm
    run_name = build_run_name(args, use_separate_models)
    root_path = f"{args.output_dir}/{args.task}/{run_name}"
    os.makedirs(root_path, exist_ok=True)

    generic_prompt = build_generic_prompt(task_instruction, args.real_robot)
    has_test_set = dataset.len_test_positives() + dataset.len_test_negatives() > 0

    # Shared kwargs for evaluate_prompt calls
    eval_kw = dict(
        model=model, processor=processor, prompt_fn=prompt_vlm,
        critic_sys_prompt=critic_prompt, dataset=dataset,
    )

    # -------------------------------------------------------------------
    # Optimisation loop
    # -------------------------------------------------------------------
    for rep in range(args.start_index, args.start_index + args.num_repeats):
        np.random.seed(rep)
        target_path = f"{root_path}/{run_name}_{rep}"
        os.makedirs(target_path, exist_ok=True)

        best_instruction = generic_prompt

        print("INSTRUCTION:")
        print(best_instruction)
        best_stats, best_fails = evaluate_prompt(
            **eval_kw, instruction=best_instruction, val=True, debug=True,
        )
        best_score = compute_objective(best_stats)

        # Logging
        logger = TensorBoardLogger(
            target_path=target_path,
            name=f"Meta-Critic-{run_name}-#{rep}",
            args=args,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
        )
        logger.write(step=0, score=best_score, prompt=best_instruction,
                     stats=best_stats, mode="train", log_prompt=False)
        logger.write(step=0, score=best_score, prompt=best_instruction,
                     stats=best_stats, mode="best", verbose=True)

        test_instructions = best_instruction
        if has_test_set:
            test_stats, _ = evaluate_prompt(
                **eval_kw, instruction=best_instruction, val=False, debug=False,
            )
            logger.write(step=0, score=compute_objective(test_stats),
                         prompt=test_instructions, stats=test_stats,
                         mode="test", verbose=True)

        if args.plot:
            plot_examples(
                logger, model, processor, prompt_vlm, critic_prompt,
                dataset, best_instruction, step=0,
            )

        for iteration in range(1, args.max_iter + 1):
            print(f"--- Refining iteration {iteration}  current score {best_score} ---")

            random.shuffle(best_fails)
            instruction = refine_prompt(
                meta_model, meta_processor, meta_prompt_vlm,
                meta_critic_prompt, robot_desc, dataset,
                best_prompt=best_instruction, prompt_kwargs=mc_prompt_kwargs,
                stats=best_stats, fails=best_fails, debug=False,
            )

            stats, fails = evaluate_prompt(
                **eval_kw, instruction=instruction, val=True, debug=False,
            )
            score = compute_objective(stats)
            logger.write(step=iteration, score=score, prompt=instruction,
                         stats=stats, mode="train", verbose=False, log_prompt=False)

            if score > best_score:
                best_score = score
                best_instruction = instruction
                best_fails = fails.copy()
                best_stats = stats
                logger.write(step=iteration, score=best_score,
                             prompt=best_instruction, stats=best_stats,
                             mode="best", verbose=True)

            if (has_test_set
                    and iteration % args.test_every == 0
                    and test_instructions != best_instruction):
                test_stats, _ = evaluate_prompt(
                    **eval_kw, instruction=best_instruction, val=False, debug=False,
                )
                test_instructions = best_instruction
                logger.write(step=iteration, score=compute_objective(test_stats),
                             prompt=test_instructions, stats=test_stats,
                             mode="test", verbose=True)

                if args.plot:
                    plot_examples(
                        logger, model, processor, prompt_vlm, critic_prompt,
                        dataset, best_instruction, step=iteration,
                    )

        # -- end of iterations for this repeat --
        logger.write(step=args.max_iter, score=best_score,
                     prompt=best_instruction, stats=best_stats,
                     mode="best", verbose=True)

        if has_test_set:
            test_stats, test_failures = evaluate_prompt(
                **eval_kw, instruction=best_instruction, val=False, debug=False,
            )
            logger.write(step=args.max_iter, score=compute_objective(test_stats),
                         prompt=best_instruction, stats=test_stats,
                         mode="test", verbose=True)
            with open(os.path.join(target_path, "test_failure_cases.pkl"), "wb") as f:
                pickle.dump(test_failures, f)
        else:
            test_stats = {}

        results = {
            "instruction": best_instruction,
            "score": best_score,
            "stats": best_stats,
            "test_stats": test_stats,
        }

        if args.plot:
            plot_examples(
                logger, model, processor, prompt_vlm, critic_prompt,
                dataset, best_instruction, step=args.max_iter,
            )

        with open(os.path.join(target_path, "results.pkl"), "wb") as f:
            pickle.dump(results, f)
        with open(os.path.join(target_path, "val_failure_cases.pkl"), "wb") as f:
            pickle.dump(best_fails, f)

        logger.close()


if __name__ == "__main__":
    main()
