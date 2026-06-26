from qwen_vl_utils import process_vision_info

import torch
import torch.nn.functional as F

import numpy as np
import warnings

def _single_token_id(tokenizer, s: str):
    ids = tokenizer.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def qwen_binary_cap_prob(model, processor, messages, prompt_kwargs=None, debug=False, threshold=0.8, num_samples=10):
    samples, p0, p1 = sample_qwen_binary_N(model, processor, messages, N=num_samples, debug=debug)
    norm_p1 = p1/(p0+p1)
    if norm_p1 > threshold:
        return "1"
    return "0"

def qwen_binary_cap_samples(model, processor, messages, prompt_kwargs=None, debug=False, threshold=0.8, num_samples=10):
    samples, p0, p1 = sample_qwen_binary_N(model, processor, messages, N=num_samples, debug=debug)
    mean_samples = np.mean(samples, axis=0)
    if mean_samples > threshold:
        return "1"
    return "0"


def sample_qwen_binary_N(
    model,
    processor,
    messages,
    N,
    temperature=1.0,
    debug=False,
):
    """
    Strict binary sampler with graceful fallback:
    - Prefer: next token is EXACTLY "0" or "1" (single token, no whitespace)
    - If not, do NOT raise; instead warn and return fallback_value for all N samples.
    - Returns: (outs, p0, p1)
      where p0/p1 are model probs for token "0"/"1" at that next-token step.
    """
    tok = processor.tokenizer
    id0 = _single_token_id(tok, "0")
    id1 = _single_token_id(tok, "1")
    if id0 is None or id1 is None:
        raise ValueError(
            f"Tokenizer does not encode '0'/'1' as single tokens: id0={id0}, id1={id1}."
        )

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    image_inputs = image_inputs or None
    video_inputs = video_inputs or None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model(**inputs, use_cache=True)
        logits = out.logits[0, -1, :]  # next-token logits

        if temperature and temperature != 1.0:
            logits = logits / float(temperature)

        probs = F.softmax(logits, dim=-1)

        # "True" next-token probs for the digit tokens (not renormalized)
        p0 = probs[id0]
        p1 = probs[id1]

        top_id = int(torch.argmax(probs).item())
        if top_id not in (id0, id1):

            # Keep this lightweight but informative
            msg = (
                f"[WARN] Non-binary next token {tok.decode([top_id])!r}; "
                f"falling back to 0. "
                f"(p0={p0.item():.6f}, p1={p1.item():.6f})"
            )
            if debug:
                topv, topi = torch.topk(probs, 10)
                top_tokens = [tok.decode([i.item()]) for i in topi]
                msg += f" top10={top_tokens}"
            warnings.warn(msg)

            outs = [0] * N
            return outs, 1.0, 0.0

        # If we are here, top-1 is binary; sample from renormalized {0,1}
        p01 = torch.stack([p0, p1])
        p01 = p01 / p01.sum()

        samples = torch.multinomial(p01, num_samples=N, replacement=True)
        outs = [0 if s.item() == 0 else 1 for s in samples]

    if debug:
        # Show both renormalized and raw probs if you like
        print(
            f"renorm p(0)={p01[0].item():.4f}, renorm p(1)={p01[1].item():.4f} | "
            f"raw p0={p0.item():.6f}, raw p1={p1.item():.6f} | samples[:10]={outs[:10]}"
        )

    return outs, p0, p1



def get_qwen3_32b():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-32B-Instruct",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct", use_fast=True)
    return model, processor

def get_qwen3_8b():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", use_fast=True)
    return model, processor

def prompt_qwen(model, processor, messages, prompt_kwargs=None, debug=True):

    if prompt_kwargs is None:
        prompt_kwargs = dict(max_new_tokens=200, do_sample=True, top_p=0.9, top_k=50, temperature=0.7)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    # Qwen processor is happier with None than empty lists
    image_inputs = image_inputs or None
    video_inputs = video_inputs or None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    eos = processor.tokenizer.eos_token_id
    pad = processor.tokenizer.pad_token_id
    if eos is None:
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if pad is None:
        pad = eos
    gen_kwargs = dict(
        eos_token_id=eos,
        pad_token_id=pad,
        use_cache=True,
    )
    gen_kwargs.update(prompt_kwargs)

    with torch.inference_mode():
        generation = model.generate(**inputs, **gen_kwargs)
    raw_output = processor.tokenizer.decode(generation[0], skip_special_tokens=True)

    if debug:
        print("RAW OUTPUT:")
        print(raw_output)
        print("---")

    split_token = "assistant"
    if split_token in raw_output:
        output_text = raw_output.split(split_token, 1)[-1].strip()
    else:
        output_text = raw_output.strip()
    return output_text