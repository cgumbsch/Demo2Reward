

EXPLORATION = dict(max_new_tokens=1000, do_sample=True, temperature=1.3, top_p=0.97, top_k=50)
DEFAULT = dict(max_new_tokens=1000, do_sample=True, temperature=0.9, top_p=0.8, top_k=50)
BALANCED = dict(max_new_tokens=1000, do_sample=True, temperature=1.0, top_p=0.92,  top_k=100)
prompt_strategy = {'explore': EXPLORATION, 'balanced': BALANCED,  'default': DEFAULT,}


class SamplingSchedule:
    """
    Linear exploration schedule for LLM/VLM sampling parameters.

    Each step() moves parameters by 1/num_steps of the full range
    from start -> max. After num_steps calls, values are clamped at max.
    """

    def __init__(
        self,
        *,
        max_new_tokens=1000,
        do_sample=True,

        # temperature
        temperature_start=1.0,
        temperature_max=2.0,

        # nucleus sampling
        top_p_start=0.93,
        top_p_max=0.98,

        # top-k sampling
        top_k_start=100,
        top_k_max=400,

        num_steps=100,
    ):
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.num_steps = num_steps

        # store ranges
        self._t_start = temperature_start
        self._t_max = temperature_max

        self._p_start = top_p_start
        self._p_max = top_p_max

        self._k_start = top_k_start
        self._k_max = top_k_max

        # precompute increments
        self._t_step = (temperature_max - temperature_start) / num_steps
        self._p_step = (top_p_max - top_p_start) / num_steps
        self._k_step = (top_k_max - top_k_start) / num_steps

        self.reset()

    def step(self):
        """Increase parameters by one step toward their max values."""
        self.temperature = min(self._t_max, self.temperature + self._t_step)
        self.top_p = min(self._p_max, self.top_p + self._p_step)
        self.top_k = min(self._k_max, self.top_k + self._k_step)

    def reset(self):
        """Reset parameters to their initial values."""
        self.temperature = self._t_start
        self.top_p = self._p_start
        self.top_k = self._k_start

    def as_dict(self):
        """Return current sampling parameters as a dict."""
        return dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            top_k=int(round(self.top_k)),
        )


def print_messages(messages):
    for msg in messages:
        print(f"\n=== {msg['role'].upper()} ===")
        for block in msg["content"]:
            if block["type"] == "text":
                print(block["text"])
            elif block["type"] == "image":
                print("[IMAGE]")


def extract_answer_after_substring(text: str, substring="Final Instruction:"):
    s = text.strip()
    # look for the last substring (in case the prompt is echoed)
    idx = s.rfind(substring)
    if idx == -1:
        # fallback: first non-empty line
        first = next((ln for ln in s.splitlines() if ln.strip()), "")
        return first.strip()
    # extract everything after the substring (including multiple lines)
    return s[idx + len(substring):].strip()


def get_vlm_funcs(name: str):
    """Return ``(model, processor, prompt_fn)`` for a supported VLM backbone."""
    if name == 'qwen3_8b':
        from vlms.qwen_utils import get_qwen3_8b, prompt_qwen
        model, processor = get_qwen3_8b()
        prompt_vlm = prompt_qwen
    elif name == 'qwen3_32b':
        from vlms.qwen_utils import get_qwen3_32b, prompt_qwen
        model, processor = get_qwen3_32b()
        prompt_vlm = prompt_qwen
    else:
        raise ValueError(f"Unknown VLM: {name}")
    return model, processor, prompt_vlm