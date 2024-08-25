import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from .tokenizer import Tokenizer


def run_logit_lens(
    hidden_outputs: list[Tensor], norm: nn.Module, logit_lens: nn.Module
) -> tuple[Tensor, Tensor]:
    tokens = []
    probs = []
    for hidden_output in hidden_outputs:
        hidden_output = hidden_output[:, -1]
        logits = logit_lens(norm(hidden_output))
        prob = torch.softmax(logits, dim=-1)[0, :].cpu()
        token = logits.argmax(-1).cpu()
        prob_t = prob[token]
        probs.append(prob_t)
        tokens.append(token)
    return torch.tensor(tokens), torch.tensor(probs)


def run_logit_lens_all(
    hidden_outputs: list[Tensor], norm: nn.Module, logit_lens: nn.Module
) -> tuple[Tensor, Tensor]:
    all_tokens = []
    all_probs = []
    for hidden_output in hidden_outputs:
        logits = logit_lens(norm(hidden_output))
        probs = torch.softmax(logits, dim=-1)[0, :, :].cpu()
        tokens = logits.argmax(-1).cpu()
        probs_t = probs[torch.arange(tokens.shape[1]), tokens[0, :]]
        all_probs.append(probs_t)
        all_tokens.append(tokens[0, :])
    return torch.stack(all_tokens), torch.stack(all_probs)


def process_attn(
    attn: list[Tensor]
) -> list[np.ndarray]:
    all_attns = []
    for attn_i in attn:
        attn_i_np = attn_i.float().cpu().numpy()
        all_attns.append(attn_i_np[0, :, :, :].max(axis=0))

    return all_attns


@torch.inference_mode()
def generate_with_logit_lens(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt_tokens: list[list[int]],
    max_gen_len: int,
    temperature: float,
    top_p: float = 0.9,
    device: str = "cpu",
):
    norm = model.norm
    logit_lens = model.output
    params = model.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)


    prev_pos = 0
    eot_reached = torch.tensor([False] * bsz, device=device)
    input_text_mask = tokens != pad_id

    all_hidden_tokens = []
    all_hidden_probs = []
    all_attentions = []
    if min_prompt_len == total_len:
        logits, hidden_outputs, attn = model.forward(tokens, prev_pos, return_hidden=True, return_attn=True)
        tokens, probs = run_logit_lens(hidden_outputs, norm,logit_lens)
        str_tokens = [tokenizer.decode([t]) for t in tokens.tolist()]
        all_hidden_tokens.append(str_tokens)
        all_hidden_probs.append(probs.tolist())
        all_attentions.append(process_attn(attn))

    for cur_pos in range(min_prompt_len, total_len):
        logits, hidden_outputs, attn = model.forward(
            tokens[:, prev_pos:cur_pos], prev_pos, return_hidden=True, return_attn=True,
        )
        hidden_tokens, probs = run_logit_lens_all(hidden_outputs, norm, logit_lens)
        str_hidden_tokens = [[tokenizer.decode([t]) for t in seq] for seq in hidden_tokens.tolist()]
        all_hidden_tokens.append(str_hidden_tokens)
        all_hidden_probs.append(probs.tolist())
        all_attentions.append(process_attn(attn))

        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        eot_reached |= (~input_text_mask[:, cur_pos]) & (
            next_token == tokenizer.eot_id
        )
        prev_pos = cur_pos
        if all(eot_reached):
            break

    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
        start = len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        probs = None

        # cut to eot tok if any
        if tokenizer.eot_id in toks:
            eot_idx = toks.index(tokenizer.eot_id)
            toks = toks[:eot_idx]
        out_tokens.append(toks)
        out_logprobs.append(probs)

    return out_tokens, all_hidden_tokens, all_hidden_probs, all_attentions


def sample_top_p(probs: Tensor, p: float) -> Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


@torch.inference_mode()
def generate(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt_tokens: list[list[int]],
    max_gen_len: int,
    temperature: float,
    top_p: float = 0.9,
    logprobs: bool = False,
    echo: bool = False,
    device: str = "cpu",
) -> tuple[list[int], list[float] | None]:
    params = model.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

    prev_pos = 0
    eot_reached = torch.tensor([False] * bsz, device=device)
    input_text_mask = tokens != pad_id
    if min_prompt_len == total_len:
        logits = model.forward(tokens, prev_pos)
        token_logprobs = -F.cross_entropy(
            input=logits.transpose(1, 2),
            target=tokens,
            reduction="none",
            ignore_index=pad_id,
        )

    for cur_pos in range(min_prompt_len, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        if logprobs:
            token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens[:, prev_pos + 1 : cur_pos + 1],
                reduction="none",
                ignore_index=pad_id
            )
        eot_reached |= (~input_text_mask[:, cur_pos]) & (
            next_token == tokenizer.eot_id
        )
        prev_pos = cur_pos
        if all(eot_reached):
            break

    if logprobs:
        token_logprobs = token_logprobs.tolist()
    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
        start = 0 if echo else len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        probs = None
        if logprobs:
            probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]

        # cut to eot tok if any
        if tokenizer.eot_id in toks:
            eot_idx = toks.index(tokenizer.eot_id)
            toks = toks[:eot_idx]
            probs = probs[:eot_idx] if logprobs else None
        out_tokens.append(toks)
        out_logprobs.append(probs)

    return (out_tokens, out_logprobs if logprobs else None)
