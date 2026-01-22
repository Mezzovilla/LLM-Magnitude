# Magnitude 

## Configuration

### Hugging Face Token

Check [this tutorial](https://www.youtube.com/watch?v=1h6lfzJ0wZw&t=732s) for creating a token.

### Download the model

Here we've tryed using [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B).

Choose the model name as the `checkpoint` variable:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "HuggingFaceTB/SmolLM2-135M"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint)
```

After reading the model, install it in the local package:

```python
from pathlib import Path
project_path = Path().cwd()

model_path = project_path / f"models/{checkpoint}"

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
```

