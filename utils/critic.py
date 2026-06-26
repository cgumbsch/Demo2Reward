"""Core Demo2Reward critic evaluation and prompt refinement."""

import time
import random

from vlms.vlm_utils import print_messages, extract_answer_after_substring
from utils.prompts import build_critic_user_prompt, build_refine_intro


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_binary(text: str) -> int:
    """Extract a single-character 0/1 response; returns -1 on failure."""
    stripped = text.strip()
    if len(stripped) != 1:
        return -1
    return stripped[0]


def _parse_critic_output(text: str) -> str:
    return extract_answer_after_substring(text, substring="Task completion:")


# ---------------------------------------------------------------------------
# Single-video critic call
# ---------------------------------------------------------------------------

def eval_single(model, processor, prompt_fn, critic_sys_prompt,
                prompt, frames, debug):
    """Run the critic VLM on one video and return its raw text output."""
    start = time.perf_counter()

    content = []
    for frame in frames:
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": critic_sys_prompt}]},
        {"role": "user", "content": content},
    ]
    kwargs = dict(max_new_tokens=5, do_sample=False, top_p=1.0, top_k=0, temperature=0)
    raw_output = prompt_fn(
        model=model, processor=processor, messages=messages,
        debug=debug, prompt_kwargs=kwargs,
    )

    if debug:
        elapsed = time.perf_counter() - start
        print(f".......... Elapsed time: {elapsed:.3f} .................")
    return raw_output


# ---------------------------------------------------------------------------
# Batch evaluation helpers
# ---------------------------------------------------------------------------

def _eval_split(model, processor, prompt_fn, critic_sys_prompt, prompt,
                videos_iter, label, debug):
    """Score every video from *videos_iter* and collect failure cases.

    Returns ``(correct, incorrect, failure_cases)`` where *correct* and
    *incorrect* are counts.
    """
    correct = 0
    incorrect = 0
    failure_cases = []
    expected = 1 if label == "positive" else 0

    if debug:
        print(f"{'Successful' if expected == 1 else 'Unsuccessful'} frames:")

    for video in videos_iter():
        raw = eval_single(
            model, processor, prompt_fn, critic_sys_prompt,
            prompt, video, debug,
        )
        reward = int(_parse_binary(raw))
        if reward == expected:
            correct += 1
        else:
            incorrect += 1
            failure_cases.append({
                "image": video[-1],
                "output": _parse_critic_output(raw),
                "video": video,
            })
        if debug:
            tag = "Success" if expected == 1 else "Fail"
            print(f"------- {tag} Response: {reward} ------------")
            print(raw)
            print("Provided Critic Response:", _parse_critic_output(raw))
            print("-------------------------------------------------------------------")

    return correct, incorrect, failure_cases


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_prompt(model, processor, prompt_fn, critic_sys_prompt,
                    dataset, instruction, val, debug=False):
    """Evaluate *instruction* on the val or test split.

    Returns ``(stats, failure_cases)``.
    """
    if val:
        num_pos = dataset.len_val_positives()
        num_neg = dataset.len_val_negatives()
        pos_videos = dataset.get_val_positive_videos
        neg_videos = dataset.get_val_negative_videos
    else:
        num_pos = dataset.len_test_positives()
        num_neg = dataset.len_test_negatives()
        pos_videos = dataset.get_test_positive_videos
        neg_videos = dataset.get_test_negative_videos

    prompt = build_critic_user_prompt(instruction)

    tp, fn, pos_fails = _eval_split(
        model, processor, prompt_fn, critic_sys_prompt,
        prompt, pos_videos, "positive", debug,
    )
    tn, fp, neg_fails = _eval_split(
        model, processor, prompt_fn, critic_sys_prompt,
        prompt, neg_videos, "negative", debug,
    )
    failure_cases = pos_fails + neg_fails

    stats = {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "positives": num_pos, "negatives": num_neg,
        "TPR": tp / num_pos * 100, "FNR": fn / num_pos * 100,
        "TNR": tn / num_neg * 100, "FPR": fp / num_neg * 100,
    }
    if debug:
        print(f"True Positive Rate {stats['TPR']:.1f}%")
        print(f"True Negative Rate {stats['TNR']:.1f}%")
        print(f"False Positive Rate {stats['FPR']:.1f}%")
        print(f"False Negative Rate {stats['FNR']:.1f}%")

    return stats, failure_cases


def eval_videos(model, processor, prompt_fn, critic_sys_prompt,
                instruction, videos, debug=False):
    """Score a flat list of videos and return a list of binary rewards."""
    prompt = build_critic_user_prompt(instruction)
    rewards = []
    for video in videos:
        raw = eval_single(
            model, processor, prompt_fn, critic_sys_prompt,
            prompt, video, debug,
        )
        rewards.append(int(_parse_binary(raw)))
    return rewards


def plot_examples(logger, model, processor, prompt_fn, critic_sys_prompt,
                  dataset, instruction, step):
    """Log example prediction-vs-GT plots for val and test splits."""
    examples = [
        (dataset.get_val_positive_example_video,  "val_pos_ex_rewards",  "best"),
        (dataset.get_val_negative_example_video,   "val_neg_ex_rewards",  "best"),
        (dataset.get_test_positive_example_video,  "test_pos_ex_rewards", "test"),
        (dataset.get_test_negative_example_video,  "test_neg_ex_rewards", "test"),
    ]
    for get_fn, name, mode in examples:
        videos, successes = get_fn()
        if videos:
            print("PLOTTING vid", name)
            predictions = eval_videos(
                model, processor, prompt_fn, critic_sys_prompt,
                instruction, videos, debug=False,
            )
            logger.log_ab_plot(
                name, step=step,
                a=successes, a_label="GT success",
                b=predictions, b_label="Pred. success",
                mode=mode,
            )


def refine_prompt(model, processor, prompt_fn, meta_sys_prompt,
                  robot_desc, dataset, best_prompt, prompt_kwargs,
                  stats=None, nth_frame=3, fails=None, num_fails=5, debug=True):
    """Ask the Meta-Critic to refine the current instruction.

    Returns the new instruction string.
    """
    if fails is None:
        fails = []

    reference_demo, reference_success = dataset.get_reference()

    # -- build message content --
    content = [{"type": "text", "text": build_refine_intro(robot_desc)}]

    # Reference demonstration (sub-sampled with random offset)
    content.append({
        "type": "text",
        "text": "Ground-truth reference demonstration "
                "(each is an image followed by its ground-truth labels):",
    })
    offset = random.randint(0, nth_frame)
    for img, label in zip(reference_demo[offset::nth_frame],
                          reference_success[offset::nth_frame]):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": str(label)})

    # Current instruction
    content.append({"type": "text", "text": f"Current Critic instruction:\n{best_prompt}"})

    # Performance summary
    if stats is not None:
        perf_lines = [
            f"True Positive Rate {stats['TPR']:.1f}%",
            f"True Negative Rate {stats['TNR']:.1f}%",
            f"False Positive Rate {stats['FPR']:.1f}%",
            f"False Negative Rate {stats['FNR']:.1f}%",
        ]
        content.append({
            "type": "text",
            "text": ("Performance summary of the current Critic instruction "
                     "over evaluation episodes:\n" + "\n".join(perf_lines)),
        })

    if fails:
        content.append({
            "type": "text",
            "text": "Here are exemplar videos which the Critic judged incorrectly:",
        })
        for fail in fails[:num_fails]:
            for frame in fail["video"]:
                content.append({"type": "image", "image": frame})
            content.append({
                "type": "text",
                "text": f"Incorrect Critic output: {fail['output']}",
            })

    content.append({
        "type": "text",
        "text": ("First provide the reasoning (3-5 sentences), then provide the "
                 "instruction - both exactly as required by the output format."),
    })

    messages = [
        {"role": "system", "content": [{"type": "text", "text": meta_sys_prompt}]},
        {"role": "user", "content": content},
    ]
    if debug:
        print_messages(messages)

    raw_output = prompt_fn(
        model=model, processor=processor, messages=messages,
        debug=debug, prompt_kwargs=prompt_kwargs,
    )
    return extract_answer_after_substring(raw_output, substring="Final Instruction:")
