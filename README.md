## Llama Attention Map Viewer
This is a simple repo based off of the llama-3.1 code from [llama](https://github.com/meta-llama/llama-models/tree/main) that allows you to view the attention maps of every layer where each token is represented by it's logit lens value (the predicted token after passing the hidden layer output through the final output layer). The repo is meant to be accessible and easy to hack for your own experiments.

## Installation
You can install the requirements with the following command:
```bash
pip install -r requirements.txt
```
and requires python Python >= 3.10. You'll need to download the Llama 3.1 model weights as well as the tokenizer which can be found at [llama](https://github.com/meta-llama/llama-models/tree/main).


## Usage and Explanation

You can run the script with the following command:
```bash
python run.py --prompt "Hello!" --model_path "path/to/model" --tokenizer_path "path/to/tokenizer" --output_dir "path/to/saved/attention_maps"
```
It uses accelerate with bitsandbytes to run the model with 8 bit precision.

It will save the attention maps in the `output_dir`. The files are specified `attn_{i}_{j}.png` where `i` is the current token it's predicting. Since the attention maps are very larged they are divided up into windows of size 32 (so you are looking at attention over 32 tokens) and they move in increments of 5, the `j` represents the increment number. So for example if you are looking at the 10th predicted token and the input sequence was 32 (so there are 42 total tokens), then `attn_10_0.png` will be 0 to 32, `attn_10_1.png` will be 5 to 37, `attn_10_2.png` will be 10 to 42, etc.

Each image will be 32 in length (for the token window) and 33 in height. The color of each cell is the attention value and the token is the logit lens (predicted token after passing the hidden layer output through the final output layer). Each row represents the output for a layer and the last row, 33rd, is the final predicted output.

Below is an example of the attention maps for the propmt "Alice and Bob are siblings. Who is Bob's sister?". You can see it's decoding the `'s` after `Bob` and is attending to the 45th column which corresponds to the token `sister` in the input. The highest attended tokens are close to the token `hood` in the output space.

![Attention Maps](https://github.com/dillonalaird/llama-attn-maps/blob/main/assets/attn_17_8.png)
