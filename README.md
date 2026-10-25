# PII Masker

PII Masker is a Python tool that detects and masks personally identifiable information in text. It uses a fine-tuned DeBERTa-v3 token-classification model and returns both masked text and structured detection results.

## What it can detect

- Names
- Email addresses
- Phone numbers
- IDs and other identification numbers
- Usernames
- Personal URLs
- Street addresses
- Common SSN formats through an additional pattern check

## Project structure

```text
pii-masker/
  model.py                         # PIIMasker inference class
  output_model/deberta3base_1024/  # tokenizer files and downloaded model weights
README.md
requirements.txt
setup.py
```

## Setup for local inference

Python 3.12 or newer is recommended.

```powershell
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch transformers sentencepiece
```

Download the trained model weights. They are intentionally not stored in Git because the weights file is larger than GitHub's normal 100 MB file limit.

```powershell
hf download hydroxai/pii_model_weight --include "*.safetensors" --local-dir pii-masker\output_model\deberta3base_1024
```

If the `hf` command is not available, install the Hugging Face Hub CLI first:

```powershell
python -m pip install -U huggingface_hub
```

## Quick start

From the repository root:

```powershell
python -c "import sys; sys.path.insert(0,'pii-masker'); from model import PIIMasker; m=PIIMasker(); print(m.mask_pii('My email is alice@example.com and my SSN is 123-45-6789.'))"
```

Example output:

```text
('My email is [B-EMAIL] and my SSN is [B-SSN].',
 {'B-EMAIL': ['alice@example.com'], 'B-SSN': ['123-45-6789']})
```

The first value is the masked text. The second value is a dictionary of detected entities.

## Python usage

```python
import sys

sys.path.insert(0, "pii-masker")
from model import PIIMasker

masker = PIIMasker()
masked_text, pii_dict = masker.mask_pii(
    "John Doe can be contacted at john@example.com."
)

print(masked_text)
print(pii_dict)
```

## How it works

1. The tokenizer converts the input text into model tokens while preserving text positions.
2. DeBERTa-v3 classifies each token as ordinary text or a PII category.
3. Adjacent pieces are joined back into complete detected values.
4. Detected values are replaced with labels such as `[B-EMAIL]`.
5. The function returns the masked text and a structured dictionary.

## Important notes

- Model predictions should be tested with data that matches your use case before production use.
- Do not commit `.venv`, caches, secrets, or downloaded model weights.
- The project supports CPU inference and uses CUDA automatically when a compatible GPU is available.
- The included notebooks contain additional training and RAG-related experiments.

## Related files

- `RAG_with_pii_and_milvus.ipynb` - RAG integration example
- `train_pii_longtransformer.ipynb` - long-document model experiment
- `requirements.txt` - original broader notebook/training dependency list

