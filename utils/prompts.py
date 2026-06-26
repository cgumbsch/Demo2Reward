"""Prompt templates for Demo2Reward."""

# ---------------------------------------------------------------------------
# System prompts  — {robot_desc} is filled with "a simulated robot" or "a robot"
# ---------------------------------------------------------------------------

META_CRITIC_TEMPLATE = """\
You are the Meta-Critic, an expert at writing precise, unambiguous text instructions \
for another vision-language model ("the Critic") that must decide whether {robot_desc} \
has completed a task in a given video.

The Critic receives:
* A short sequence of RGB frames (a sparsely sampled video) from a single episode, \
ordered in time from the start to the current state (roughly 5-10 frames, every Nth frame).
* Your instruction text.

The Critic outputs task completion score as a single, binary character: "1" if the task \
is completed in the last frame; "0" otherwise. Different formatting is not allowed.

Your goal is to rewrite the Critic's instruction so that its decisions align with the \
binary ground-truth labels of success/no success.

Follow these principles at all times:

* Define success strictly by clear visual evidence in the frames of the sequence.
* Assume the video is sparsely sampled: avoid criteria that depend on seeing every \
moment of motion or exact trajectories.
* State the task goal succinctly, then specify a minimal checklist of visual cues \
required for success.
* Include explicit failure/ambiguity rules.
* Specify what to ignore.
* When needed, explicitly say how to compare frames.
* Use imperative voice, and refer to objects as they appear visually.
* Format your response exactly as follows without other additions or changes in formatting:
Reasoning: <3-5 sentences about the ambiguities or mistakes of the current prompt that \
explain the failure cases.>
Final Instruction: <updated and final instruction text>"""


CRITIC_TEMPLATE = """\
You are an expert roboticist tasked to decide whether {robot_desc} has completed a \
given task using a short video and a task instruction.

Output format:
Return a single character with no extra text:
"1" if the task is completed in this frame;
"0" otherwise.
Do not explain your answer."""


# ---------------------------------------------------------------------------
# Critic user-prompt — wraps the refined instruction for the critic
# ---------------------------------------------------------------------------

_CRITIC_USER = (
    "Here is a sequence of frames showing a robot policy attempting to solve a task. "
    "I need your help determining whether the policy is successful.\n\n"
    "Instruction: {instruction}\n"
    "Output EXACTLY a single character, either 0 or 1, to denote task completion. "
    "Use 1 if the task is completed; 0 otherwise. Use no other symbols or formatting."
)


# ---------------------------------------------------------------------------
# Zero-shot user prompt — used when no refined instruction is available
# ---------------------------------------------------------------------------

_CRITIC_USER_ZERO_SHOT = (
    "Here is a sequence of frames showing a robot policy attempting to solve a task. "
    "I need your help determining whether the policy is successful. "
    "How successfully does the robot complete the following task: {task}?\n\n"
    "Output EXACTLY a single character, either 0 or 1, to denote task completion. "
    "Use 1 if the task is completed; 0 otherwise. Use no other symbols or formatting."
)


# ---------------------------------------------------------------------------
# Meta-Critic refinement intro — the opening paragraph sent to the meta-critic
# ---------------------------------------------------------------------------

_REFINE_INTRO = (
    "Rewrite the instruction that will be given to a separate AI model (the Critic). "
    "The Critic receives a short sequence of RGB frames from a single episode and must "
    "decide whether {robot_desc} has completed the task{in_final_frame}. "
    "Your goal is to modify the instruction so that the Critic's binary decisions "
    "(1 = success, 0 = not successful) match the ground-truth labels as closely as "
    "possible. \n\n"
    "As a reference, you are given a demonstration of a successful task execution: "
    "a sequence of frames from a single episode, each followed by a binary ground-truth "
    "label indicating whether the task is already completed in that particular frame. "
    "Use this reference, together with the performance summary{and_failure_cases}, "
    "to refine the instruction."
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_prompts(real_robot: bool):
    """Return (meta_critic_sys_prompt, critic_sys_prompt, robot_desc)."""
    robot_desc = "a robot" if real_robot else "a simulated robot"
    meta = META_CRITIC_TEMPLATE.format(robot_desc=robot_desc)
    critic = CRITIC_TEMPLATE.format(robot_desc=robot_desc)
    return meta, critic, robot_desc


def build_generic_prompt(task_instruction: str, real_robot: bool) -> str:
    """Build the initial (un-refined) instruction."""
    robot_noun = "this robot" if real_robot else "this simulated robot"
    return (
        f"Evaluate the behavior of {robot_noun}. "
        f"The robot was instructed with the task: {task_instruction}."
    )


def build_critic_user_prompt(instruction: str) -> str:
    """Build the full user prompt that wraps the instruction for the critic."""
    return _CRITIC_USER.format(instruction=instruction)


def build_zero_shot_user_prompt(task: str) -> str:
    """Build a user prompt for zero-shot evaluation (no refined instruction)."""
    return _CRITIC_USER_ZERO_SHOT.format(task=task)


def build_refine_intro(robot_desc: str) -> str:
    """Build the opening paragraph for the meta-critic refinement request."""
    return _REFINE_INTRO.format(
        robot_desc=robot_desc,
        in_final_frame=" in the final frame of that sequence",
        and_failure_cases=" and failure cases",
    )
