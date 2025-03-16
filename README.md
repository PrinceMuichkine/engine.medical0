# Medical0

<div align="center">
<img src="images/Medical0.png" alt="icon" style="width:50px; vertical-align:middle;" />

**A Medical Large Vision-Language Model for Comprehension and Generation**

<a href='https://arxiv.org/abs/2502.09838'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
<a href='https://huggingface.co/lintw/Medical0-M3'><img src='https://img.shields.io/badge/Model-Huggingface-yellow'></a>
<a href='https://huggingface.co/datasets/lintw/VL-Health'><img src='https://img.shields.io/badge/Dataset-Huggingface-E59FB6'></a>

</div>

## 🌟 Overview
**Medical0** is an advanced medical Large Vision-Language Model with a unified framework that integrates both medical visual comprehension and generation capabilities. The model uses a **heterogeneous low rank adaptation (H-LoRA)** approach that enables pre-trained large language models to efficiently process and generate visual medical content.

### 📚 Capabilities
Medical0 supports **7** types of medical comprehension tasks and **5** types of medical generation tasks:

- **Comprehension Tasks**: Identify modalities, map anatomical structures, detect abnormalities, recognize congenital variants, and more
- **Generation Tasks**: Enhance image clarity, highlight abnormalities, delineate structures, and perform multi-structure enhancement

## 🛠️ Quick Setup

### One-Line Installation

```bash
# Clone and install
git clone https://github.com/PrinceMuichkine/engine.medical0
cd engine.medical0
conda create -n Medical0 python=3.10
conda activate Medical0
pip install -r requirements.txt
```

### Download Required Models and Weights

```bash
# Install HuggingFace CLI
pip install huggingface_hub
huggingface-cli login
export HUGGINGFACE_TOKEN=your_token_here

# Download models and weights
mkdir -p openai/clip-vit-large-patch14-336
cd openai/clip-vit-large-patch14-336
wget https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/config.json
wget https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/preprocessor_config.json
wget https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/pytorch_model.bin
cd ../..

mkdir -p microsoft/Phi-3-mini-4k-instruct
cd microsoft/Phi-3-mini-4k-instruct
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/config.json
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/tokenizer_config.json
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/tokenizer.model
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/pytorch_model.bin
cd ../..

mkdir -p weights
wget https://huggingface.co/lintw/Medical0-M3/resolve/main/com_hlora_weights.bin -O weights/com_hlora_weights.bin
wget https://huggingface.co/lintw/Medical0-M3/resolve/main/fusion_layer_weights.bin -O weights/fusion_layer_weights.bin
wget https://huggingface.co/lintw/Medical0-M3/resolve/main/gen_hlora_weights.bin -O weights/gen_hlora_weights.bin
wget https://huggingface.co/lintw/Medical0-L14/resolve/main/com_hlora_weights_phi4.bin -O weights/com_hlora_weights_phi4.bin
```

### Install Additional Dependencies

```bash
apt-get update
apt-get install -y libgl1-mesa-glx
pip install flask flask_cors pillow opencv-python numpy
```

## ⚡ Running the API Server

You have two options to run the server:

### Option 1: Standard Mode (load on first request)
```bash
python apy-bk.py --port 8080
```

### Option 2: Preload Mode (load models at startup)
```bash
python apy.py --port 8080 --preload-models
```

Or run with no preloading (models will not load on request):
```bash
python apy.py --port 8080 --no-preload-models
```

### Simple Start Script
Alternatively, use the start script:
```bash
chmod +x engine/start_server.sh
./engine/start_server.sh --port 8080 --no-preload
```

## 🧠 Using the API

### Analyze Images
Send POST requests to `/analyze-image` with:
- `image`: base64-encoded image or URL
- `analysis_type`: (optional) specific analysis type
- `question`: (optional) custom analysis prompt

### Reconstruct/Enhance Images
Send POST requests to `/reconstruct-image` with:
- `image`: base64-encoded image or URL
- `analysis_type`: (optional) enhancement type
- `question`: (optional) custom enhancement prompt

## 📚 Model Types

Medical0 is available in two configurations:

- **Medical0-M3**: Smaller version based on Phi-3-mini, optimized for speed and reduced memory
- **Medical0-L14**: Larger version based on Phi-4, designed for higher performance

## 🤝 Acknowledgments
This project builds upon:
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [LLaVA++](https://github.com/mbzuai-oryx/LLaVA-pp)
- [Taming Transformers](https://github.com/CompVis/taming-transformers)

## ⚖️ License
This repository is under [Apache License 2.0](LICENSE).
