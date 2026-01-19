
# Hugging Face Token

Check [this tutorial](https://www.youtube.com/watch?v=1h6lfzJ0wZw&t=732s) for creating a token.

# Download the model

Choose the model name as the `checkpoint` variable:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "Norod78/english-sienfeld-distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint)
```

After reading the model, install it in the local package:


```python
from pathlib import Path
project_path = Path().cwd().parent

model_path = project_path / f"models/{checkpoint}"

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
```

