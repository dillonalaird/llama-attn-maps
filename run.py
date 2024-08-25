import warnings
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from accelerate import init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

from llama import (
    ChatFormat,
    ModelArgs,
    SystemMessage,
    Tokenizer,
    Transformer,
    UserMessage,
    generate,
    generate_with_logit_lens,
    llama3_1_model_list,
)

DEVICE = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt",
    help="prompt you want to pass the model",
    default="Alice and Bob are siblings. Who is Bob's sister?",
)
parser.add_argument(
    "--model_path",
    help="path to model weights .pth file",
    default="Meta-Llama-3.1-8B-Instruct/original/consolidated.00.pth",
)
parser.add_argument(
    "--tokenizer_path",
    help="path to tokenizer file",
    default="Meta-Llama-3.1-8B-Instruct/original/tokenizer.model",
)
parser.add_argument(
    "--output_dir",
    help="path to save the attention plots",
    default="attn_plots",
)


def get_date():
    current_date = datetime.now()
    formatted_date = (
        current_date.strftime("%b")
        + " "
        + str(current_date.day)
        + " "
        + current_date.strftime("%Y")
    )
    return formatted_date


def fix_arrays(arr: list[list[list[str]]]) -> np.ndarray:
    first = np.array(arr[0]).T  # [N_in, L]
    second = np.array(arr[1:])[:, :, 0]  # [N_out, L]
    return np.concatenate([first, second], axis=0)  # [N_in + N_out, L]


def fix_attn(arr: list[list[np.ndarray]]) -> list[np.ndarray]:
    first = [np.array(arr[0])]

    rest = []
    for arr_i in arr[1:]:
        rest.append(np.array(arr_i).squeeze())

    return first + rest


def pad_attn(arr: list[np.ndarray]) -> list[np.ndarray]:
    arr_out = []
    for arr_i in arr:
        arr_out.append(np.concatenate([arr_i, np.zeros((1, arr_i.shape[1]))], axis=0))
    return arr_out


def plot_tokens(save_path: str, tokens: np.ndarray, probs: np.ndarray):
    probs = probs.T
    tokens = tokens.T

    plt.subplots(figsize=(18, 8))
    ax = sns.heatmap(
        probs,
        annot=tokens,
        fmt="",
        cmap="coolwarm",
        cbar=True,
        xticklabels=False,
        yticklabels=False,
        linewidths=0.5,
        linecolor="black",
        annot_kws={"size": 6},
    )

    ax.set_aspect(aspect="auto")
    plt.savefig(save_path)


def plot_attn(
    save_dir: str,
    tokens: np.ndarray,
    attns: list[np.ndarray],
    window_size: int = 32,
    step_size: int = 5,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    for i in tqdm(range(len(attns))):
        attn_i = attns[i]
        tokens_i = tokens[:, : attn_i.shape[1]]
        if window_size < tokens_i.shape[1]:
            k = 0
            for j in range(0, tokens_i.shape[1] - window_size + step_size, step_size):
                if j + window_size > tokens_i.shape[1]:
                    end = tokens_i.shape[1]
                    start = j - ((j + window_size) - tokens_i.shape[1])
                else:
                    end = j + window_size
                    start = j

                attn_ij = attn_i[:, start:end]
                tokens_ij = tokens_i[:, start:end]
                plt.figure(figsize=(28, 18))
                ax = sns.heatmap(
                    attn_ij,
                    annot=tokens_ij,
                    fmt="",
                    cmap="coolwarm",
                    cbar=True,
                    xticklabels=[str(l) for l in range(start, end)],
                    yticklabels=True,
                    linewidth=0.5,
                    linecolor="black",
                    annot_kws={"size": 6},
                    vmin=0,
                    vmax=1,
                )
                ax.set_aspect(aspect="auto")
                plt.savefig(save_dir / f"attn_{i}_{k}.png")
                plt.close()
                k += 1
        else:
            plt.figure(figsize=(28, 18))
            ax = sns.heatmap(
                attn_i,
                annot=tokens_i,
                fmt="",
                cmap="coolwarm",
                cbar=True,
                xticklabels=True,
                yticklabels=True,
                linewidth=0.5,
                linecolor="black",
                annot_kws={"size": 6},
                vmin=0,
                vmax=1,
            )
            ax.set_aspect(aspect="auto")
            plt.savefig(save_dir / f"attn_{i}.png")
            plt.close()


def run_8bit(
    prompt: str,
    args: ModelArgs,
    weights_path: str,
    tokenizer_path: str,
    default_dtype=torch.bfloat16,
    output_dir: str = "attn_plots",
):
    torch.set_default_dtype(default_dtype)

    with init_empty_weights():
        empty_model = Transformer(args)

    bnb_quantization_config = BnbQuantizationConfig(
        load_in_8bit=True,
        llm_int8_threshold=6,
        torch_dtype=default_dtype,
        skip_modules=["output"],
    )

    # can't get this working
    # bnb_quantization_config = BnbQuantizationConfig(
    #     load_in_4bit=True,
    #     torch_dtype=default_dtype,
    #     skip_modules=["output"],
    # )

    model = load_and_quantize_model(
        empty_model,
        weights_location=weights_path,
        bnb_quantization_config=bnb_quantization_config,
    )
    model.to("cuda")
    tokenizer = Tokenizer(tokenizer_path)
    chat_format = ChatFormat(tokenizer)

    in_tokens = chat_format.encode_dialog_prompt(
        [
            SystemMessage(
                content=f"Cutting Knowledge Date: December 2023\nToday Date: {get_date()}\n\nYou are a helpful assistant"
            ),
            UserMessage(content=prompt),
        ]
    ).tokens

    # if you want to just generate some tokens
    # out_tokens, _ = generate(
    #     model,
    #     tokenizer,
    #     [in_tokens],
    #     max_gen_len=256,
    #     temperature=0.1,
    #     device="cuda",
    # )

    out_tokens, tokens, probs, attns = generate_with_logit_lens(
        model,
        tokenizer,
        [in_tokens],
        max_gen_len=256,
        temperature=0.1,
        device="cuda",
    )
    print(tokenizer.decode(out_tokens[0]))

    # Let the number of input tokens be N_in, number of output tokens be N_out and
    # number of layers be L. `tokens` will be of length N_out + 1 and each element of
    # `tokens` will be of length L and represents the intermediate tokens for each layer
    # after taking the logit lens. The first element of `tokens` is an L x N_in matrix,
    # so it represents the entire input token sequence across every layer. Every other
    # element is an L length array representing the output token for each layer (after
    # using the logit lens).

    # fix_arrays will turn tokens into an [N_in + N_out, L] matrix.
    tokens = fix_arrays(tokens)
    probs = fix_arrays(probs)

    # for plotting just the logit lens tokens
    # plot_tokens("save.png", tokens[-(len(out_tokens[0]) + 1):, :], probs[-(len(out_tokens[0]) + 1):, :])

    # create an [N_in + N_out] array of input tokens and output tokens
    all_tokens = [tokenizer.decode([t]) for t in (in_tokens + out_tokens[0])]
    # prepend the [N_in + N_out] array to tokens, so now it's [N_in + N_out, L + 1]
    # where the additional 1 is for the all the inputs (including previous outputs).
    # It's then transposed to [L + 1, N_in + N_out] to line up with the attn maps.
    all_tokens = np.concatenate([np.array(all_tokens)[:, None], tokens], axis=1).T

    # attns is an N_out + 1 length array where the first element is [L, N_in, N_in]
    # which represents all the attention maps per layer for in the input tokens. The
    # remaining N_out tokens are [L, N_in + n'] where n' goes from 1 to N_out so these
    # represent the attention maps as it's decoding.
    attns = fix_attn(attns)[1:]
    # create an L+1 layer of just 0's so we can visaulize the final predicted tokens
    attns = pad_attn(attns)

    # supress warnings from not being able to print strange tokens
    with warnings.catch_warnings(action="ignore"):
        plot_attn(output_dir, all_tokens, attns)


if __name__ == "__main__":
    args = parser.parse_args()
    models = llama3_1_model_list()
    llama3_1_8b = models[0]
    llama3_1_8b_args = ModelArgs(**llama3_1_8b.model_args)
    run_8bit(
        args.prompt,
        llama3_1_8b_args,
        args.model_path,
        args.tokenizer_path,
        args.output_dir,
    )
