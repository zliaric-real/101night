---
title: SSSM Sleep Event Detection
emoji: 💤
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.28.0
python_version: 3.10
app_file: app.py
pinned: false
---

### Hugging Face Space: SSSM Sleep Event Detection

This repo is configured to run as a Hugging Face Space using Gradio.

What's included:
- `app.py`: Gradio UI to run inference with the bundled models in `sssm/saved_models/`.
- `requirements.txt`: Dependencies for the Space.
- `runtime.txt`: Python runtime pin.

How to run locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python app.py
```

Deploy to Hugging Face Spaces (via CLI):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token YOUR_TOKEN

# Create a new Space (Gradio SDK)
huggingface-cli repo create sssm-demo --type space --sdk gradio --yes

git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/sssm-demo
git add .
git commit -m "Add Gradio app for SSSM"
git push -u origin HEAD
```

Usage notes:
- If no CSV is provided, the app generates a synthetic signal for demonstration.
- CSV should contain at least one numeric column with length ≥ 300 samples.
- Use the "Model file" dropdown to select among packaged checkpoints.

