---
library_name: transformers
language:
- en
---
# Healthcare Domain-Specific Assistant

This is a lightweight, domain-specific medical assistant fine-tuned on the **TinyLlama-1.1B** architecture using **LoRA (Low-Rank Adaptation)**. This model is designed to answer medical questions with improved accuracy and domain knowledge while remaining efficient enough to run on consumer hardware (e.g., Google Colab T4 GPU).

## Project Overview

- **Base Model**: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Fine-Tuning Technique**: QLoRA (4-bit Quantization + LoRA)
- **Dataset**: [Medical Meadow Medical Flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
- **Task**: Medical Question Answering / Instruction Following
- **Frameworks**: PyTorch, Transformers, PEFT, TRL, BitsAndBytes

## Features

- **Domain Expertise**: Fine-tuned on 33k+ medical Q&A pairs (subset used for efficiency).
- **Efficient**: Uses 4-bit quantization to fit within 15GB GPU memory.
- **Interactive UI**: Includes a Gradio chat interface for easy interaction.
- **Deployable**: Ready for deployment on HuggingFace Spaces.

## Installation

To run this project locally or in Colab, install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers>=4.36.0 datasets>=2.16.0 accelerate>=0.25.0
pip install peft>=0.7.0 bitsandbytes>=0.41.0 trl>=0.7.0
pip install evaluate rouge-score nltk gradio
```

## Usage

### 1. Training the Model
Run the provided Jupyter Notebook/Python script to fine-tune the model. Key steps include:
- Loading and preprocessing the medical flashcards dataset.
- Configuring LoRA adapters (Rank=16, Alpha=32).
- Training with `Trainer` API using 4-bit precision.

### 2. Inference
Load the fine-tuned model and generate responses:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base Model
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_model_id = "musembii/medical-assistant"  

# Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(adapter_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load Adapter
model = PeftModel.from_pretrained(base_model, adapter_model_id)

# Chat
prompt = "What are the symptoms of pneumonia?"
formatted_prompt = f"<|system|>\nYou are a helpful medical assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"

inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

##  Evaluation

The model was evaluated using quantitative metrics (BLEU, ROUGE) and qualitative human assessment.

| Metric | Score |
|--------|-------|
| BLEU | ~0.26 |
| ROUGE-1| ~0.067|
| ROUGE-L| ~0.059|

*Note: Metrics are based on a test split of the medical flashcards dataset.*

##  Deployment

The model is deployed using **Gradio** and hosted on **HuggingFace Spaces**.

- **Live Demo**: (https://huggingface.co/spaces/musembii/medical-assistant-demo)
- **Model Weights**: musembii/medical-assistant

##  Disclaimer

**This AI model is for educational and research purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.

##  Acknowledgments

- [TinyLlama Project](https://github.com/jzhang38/TinyLlama)
- [MedAlpaca](https://github.com/kbressem/medAlpaca) for the dataset.
- HuggingFace for the ecosystem.