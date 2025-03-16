# Medical0

<div align="center">
<img src="images/medical0.png" alt="icon" style="width:50px; vertical-align:middle;" />

**A Medical Large Vision-Language Model for Comprehension and Generation**
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
git clone https://github.com/PrinceMuichkine/engine
cd engine
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