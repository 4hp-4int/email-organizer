---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M/blob/main/LICENSE
language:
- en
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-14B
tags:
- chat
library_name: transformers
---

# Qwen2.5-14B-Instruct-1M
<a href="https://chat.qwenlm.ai/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/%F0%9F%92%9C%EF%B8%8F%20Qwen%20Chat%20-536af5" style="display: inline-block; vertical-align: middle;"/>
</a>

## Introduction

Qwen2.5-1M is the long-context version of the Qwen2.5 series models, supporting a context length of up to 1M tokens. Compared to the Qwen2.5 128K version, Qwen2.5-1M demonstrates significantly improved performance in handling long-context tasks while maintaining its capability in short tasks.

The model has the following features:
- Type: Causal Language Models
- Training Stage: Pretraining & Post-training
- Architecture: transformers with RoPE, SwiGLU, RMSNorm, and Attention QKV bias
- Number of Parameters: 14.7B
- Number of Paramaters (Non-Embedding): 13.1B
- Number of Layers: 48
- Number of Attention Heads (GQA): 40 for Q and 8 for KV
- Context Length: Full 1,010,000 tokens and generation 8192 tokens
  - We recommend deploying with our custom vLLM, which introduces sparse attention and length extrapolation methods to ensure efficiency and accuracy for long-context tasks. For specific guidance, refer to [this section](#processing-ultra-long-texts).
  - You can also use the previous framework that supports Qwen2.5 for inference, but accuracy degradation may occur for sequences exceeding 262,144 tokens.

For more details, please refer to our [blog](https://qwenlm.github.io/blog/qwen2.5-1m/), [GitHub](https://github.com/QwenLM/Qwen2.5), [Technical Report](https://huggingface.co/papers/2501.15383), and [Documentation](https://qwen.readthedocs.io/en/latest/).

## Requirements

The code of Qwen2.5 has been in the latest Hugging face `transformers` and we advise you to use the latest version of `transformers`.

With `transformers<4.37.0`, you will encounter the following error:
```
KeyError: 'qwen2'
```

## Quickstart

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-14B-Instruct-1M"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### Processing Ultra Long Texts

To enhance processing accuracy and efficiency for long sequences, we have developed an advanced inference framework based on vLLM, incorporating sparse attention and length extrapolation. This approach significantly improves model generation performance for sequences exceeding 256K tokens and achieves a 3 to 7 times speedup for sequences up to 1M tokens.

Here we provide step-by-step instructions for deploying the Qwen2.5-1M models with our framework.

#### 1. System Preparation

To achieve the best performance, we recommend using GPUs with Ampere or Hopper architecture, which support optimized kernels.

Ensure your system meets the following requirements:

- **CUDA Version**: 12.1 or 12.3
- **Python Version**: >=3.9 and <=3.12

**VRAM Requirements:**

- For processing 1 million-token sequences:
  - **Qwen2.5-7B-Instruct-1M**: At least 120GB VRAM (total across GPUs).
  - **Qwen2.5-14B-Instruct-1M**: At least 320GB VRAM (total across GPUs).

If your GPUs do not have sufficient VRAM, you can still use Qwen2.5-1M for shorter tasks.

#### 2. Install Dependencies

For now, you need to clone the vLLM repository from our custom branch and install it manually. We are working on getting our branch merged into the main vLLM project.

```bash
git clone -b dev/dual-chunk-attn git@github.com:QwenLM/vllm.git
cd vllm
pip install -e . -v
```


#### 3. Launch vLLM

vLLM supports offline inference or launch an openai-like server.

**Example of Offline Inference**

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct-1M")

# Pass the default decoding hyperparameters of Qwen2.5-14B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# Input the model name or path. See below for parameter explanation (after the example of openai-like server).
llm = LLM(model="Qwen/Qwen2.5-14B-Instruct-1M",
    tensor_parallel_size=4,
    max_model_len=1010000,
    enable_chunked_prefill=True,
    max_num_batched_tokens=131072,
    enforce_eager=True,
    # quantization="fp8", # Enabling FP8 quantization for model weights can reduce memory usage.
)

# Prepare your prompts
prompt = "Tell me something about large language models."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

**Example of Openai-like Server**

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-1M \
  --tensor-parallel-size 4 \
  --max-model-len 1010000 \
  --enable-chunked-prefill --max-num-batched-tokens 131072 \
  --enforce-eager \
  --max-num-seqs 1

# --quantization fp8  # Enabling FP8 quantization for model weights can reduce memory usage.
```

Then you can use curl or python to interact with the deployed model.

**Parameter Explanations:**

- **`--tensor-parallel-size`**
  - Set to the number of GPUs you are using. Max 4 GPUs for the 7B model, and 8 GPUs for the 14B model.
  
- **`--max-model-len`**
  - Defines the maximum input sequence length. Reduce this value if you encounter Out of Memory issues.

- **`--max-num-batched-tokens`**
  - Sets the chunk size in Chunked Prefill. A smaller value reduces activation memory usage but may slow down inference. 
  - Recommend 131072 for optimal performance.

- **`--max-num-seqs`**
  - Limits concurrent sequences processed. 

You can also refer to our [Documentation](https://qwen.readthedocs.io/en/latest/deployment/vllm.html) for usage of vLLM.

#### Troubleshooting:

1. Encountering the error: "The model's max sequence length (xxxxx) is larger than the maximum number of tokens that can be stored in the KV cache."

    The VRAM reserved for the KV cache is insufficient. Consider reducing the ``max_model_len`` or increasing the ``tensor_parallel_size``. Alternatively, you can reduce ``max_num_batched_tokens``, although this may significantly slow down inference.

2. Encountering the error: "torch.OutOfMemoryError: CUDA out of memory."

    The VRAM reserved for activation weights is insufficient. You can try setting ``gpu_memory_utilization`` to 0.85 or lower, but be aware that this might reduce the VRAM available for the KV cache.

3. Encountering the error: "Input prompt (xxxxx tokens) + lookahead slots (0) is too long and exceeds the capacity of the block manager."

    The input is too lengthy. Consider using a shorter sequence or increasing the ``max_model_len``.

## Evaluation & Performance

Detailed evaluation results are reported in this [📑 blog](https://qwenlm.github.io/blog/qwen2.5-1m/) and our [technical report](https://arxiv.org/abs/2501.15383).

## Citation

If you find our work helpful, feel free to give us a cite.

```
@misc{qwen2.5-1m,
    title = {Qwen2.5-1M: Deploy Your Own Qwen with Context Length up to 1M Tokens},
    url = {https://qwenlm.github.io/blog/qwen2.5-1m/},
    author = {Qwen Team},
    month = {January},
    year = {2025}
}

@article{qwen2.5,
      title={Qwen2.5-1M Technical Report}, 
      author={An Yang and Bowen Yu and Chengyuan Li and Dayiheng Liu and Fei Huang and Haoyan Huang and Jiandong Jiang and Jianhong Tu and Jianwei Zhang and Jingren Zhou and Junyang Lin and Kai Dang and Kexin Yang and Le Yu and Mei Li and Minmin Sun and Qin Zhu and Rui Men and Tao He and Weijia Xu and Wenbiao Yin and Wenyuan Yu and Xiafei Qiu and Xingzhang Ren and Xinlong Yang and Yong Li and Zhiying Xu and Zipeng Zhang},
      journal={arXiv preprint arXiv:2501.15383},
      year={2025}
}
```
