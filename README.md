# RL_DPO_implementation
This repository is the implementation of DPO Technique

## What is DPO ?
Direct Preference Optimization (DPO) is a machine learning technique used to optimize a system based on explicit user feedback or preferences. It is a type of preference-based learning where the user directly expresses their preferences between different options or outcomes.

## Finetuning TheBloke/OpenHermes-2-Mistral-7B-GPTQ using DPO

This repository contains code for finetuning the TheBloke/OpenHermes-2-Mistral-7B-GPTQd on HuggingFaceH4/ultrafeedback_binarized dataset, which contains the corresponding chosen and rejected responses for the given prompt

### Libraries Used
- `transformers`: For utilizing and fine-tuning the Roberta-base model.
- `huggingface-hub`: For accessing the model and tokenizer from the Hugging Face model hub.
- `peft`: for model pruning and quantization
- `bitsandbytes`: efficient memory usage and computation
- `optimum`:  for optimizing deep learning models
- `auto-gptq`: for automatic mixed precision training and quantization
- `trl`: for building and training recurrent neural networks
- `datasets`: For handling and processing the dataset.
- `numpy`: For numerical computations.
- `torch`: For building and training neural networks.

### Training Details

#### Hyperparameters
- learning_rate: 0.0002
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 2
- training_steps: 50
- mixed_precision_training: Native AMP


### Usage 

`
from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig
from transformers import AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("likhith231/openhermes-mistral-dpo-gptq")

inputs = tokenizer("""I have dropped my phone in water. Now it is not working what should I do now?""", return_tensors="pt").to("cuda")

model = AutoPeftModelForCausalLM.from_pretrained(
    "likhith231/openhermes-mistral-dpo-gptq",
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cuda")

generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.1,
    max_new_tokens=256,
    pad_token_id=tokenizer.eos_token_id
)

outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

`