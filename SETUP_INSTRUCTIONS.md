# HealthGPT Integration Setup Instructions

This document provides detailed instructions for setting up the HealthGPT integration with the Medical0 AI chatbot application.

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker and Docker Compose
- Git

## Setup Steps

### 1. Download Model Weights

Before running the integration, you need to download the following model weights:

#### Visual Encoder
- **Model**: `clip-vit-large-patch14-336` 
- **Source**: [Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14-336)
- **Local Path**: `./clip-vit-large-patch14-336/`

#### Base LLM Models
- **For HealthGPT-M3**: 
  - **Model**: `Phi-3-mini-4k-instruct`
  - **Source**: [Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
  - **Local Path**: `./Phi-3-mini-4k-instruct/`

- **For HealthGPT-L14**:
  - **Model**: `phi-4`
  - **Source**: [Hugging Face](https://huggingface.co/microsoft/phi-4)
  - **Local Path**: `./phi-4/`

#### HealthGPT Weights
- **H-LoRA and Fusion Layer Weights**:
  - **Source**: [HealthGPT-M3 Hugging Face](https://huggingface.co/lintw/HealthGPT-M3) and [HealthGPT-L14 Hugging Face](https://huggingface.co/lintw/HealthGPT-L14)
  - **Files Needed**:
    - `com_hlora_weights.bin` - For comprehension tasks (M3)
    - `gen_hlora_weights.bin` - For generation tasks (M3)
    - `fusion_layer_weights.bin` - For both tasks (M3)
    - `com_hlora_weights_phi4.bin` - For comprehension tasks (L14)
  - **Local Paths**:
    - `./HealthGPT-M3/com_hlora_weights.bin`
    - `./HealthGPT-M3/gen_hlora_weights.bin`
    - `./HealthGPT-M3/fusion_layer_weights.bin`
    - `./HealthGPT-L14/com_hlora_weights_phi4.bin`

#### VQGAN Weights (For Image Generation)
- **Model**: `VQGAN OpenImages (f=8), 8192, GumbelQuantization`
- **Source**: [Heidelberg Box](https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/?p=%2F&mode=list)
- **Local Path**: `./taming_transformers/ckpt/`
- **Files Needed**:
  - `last.ckpt`
  - `model.yaml`

### 2. Update Configuration Files

The `config.py` file contains paths to the model weights. Update the paths to match where you downloaded the weights:

```python
# For HealthGPTConfig_M3_COM
model_name_or_path = "./Phi-3-mini-4k-instruct"
vit_path = "./clip-vit-large-patch14-336/"
hlora_path = "./HealthGPT-M3/com_hlora_weights.bin"

# For HealthGPTConfig_M3_GEN
model_name_or_path = "./Phi-3-mini-4k-instruct"
vit_path = "./clip-vit-large-patch14-336/"
hlora_path = "./HealthGPT-M3/gen_hlora_weights.bin"
fusion_layer_path = "./HealthGPT-M3/fusion_layer_weights.bin"

# For HealthGPTConfig_L14_COM
model_name_or_path = "./phi-4"
vit_path = "./clip-vit-large-patch14-336/"
hlora_path = "./HealthGPT-L14/com_hlora_weights_phi4.bin"
```

### 3. Run with Docker Compose

After downloading all the required model weights and updating the configuration files, you can run the API using Docker Compose:

```bash
cd engine
docker-compose up -d
```

This will start the HealthGPT API service on port 5000.

### 4. Verify the API Service

You can verify that the API service is running correctly by making a GET request to the health check endpoint:

```bash
curl http://localhost:5000/api/health
```

You should receive a response similar to:
```json
{
  "status": "healthy",
  "message": "HealthGPT API is running",
  "models": ["HealthGPT-M3-COM", "HealthGPT-M3-GEN", "HealthGPT-L14-COM"]
}
```

## Integration with Medical0 AI Chatbot

The Medical0 AI chatbot has been configured to connect to the HealthGPT API service. The integration supports:

1. **Medical Image Analysis**: Upload a medical image and ask questions about it
2. **Medical Image Generation**: Request generation of medical images based on descriptions

To enable the integration, update your `.env` file in the root directory of the Medical0 project with:

```
HEALTHGPT_API_URL=http://localhost:5000
```

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**: If you encounter GPU memory issues, try:
   - Using only one model at a time
   - Reducing batch sizes in the configuration
   - Ensuring no other GPU-intensive applications are running

2. **Model Loading Errors**: Ensure all model weights are downloaded correctly and paths in the configuration files are correct.

3. **API Connection Issues**: Make sure the API service is running and accessible from the Medical0 application.

For additional assistance, please check the logs of the Docker container:

```bash
docker-compose logs healthgpt-api
``` 