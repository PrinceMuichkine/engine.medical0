# HealthGPT Optimized Model Server

This directory contains optimized tools for running HealthGPT inferences with significantly reduced latency.

## Performance Improvements

The standard batch inference script (`com_infer.sh`) can take 3-5 minutes per image, which isn't viable for commercial applications. This optimized version:

1. Loads the model once and keeps it in memory (no more 200+ second model loading times)
2. Provides an API server for fast inference requests (30-40 seconds per image)
3. Caches processed images to avoid redundant processing
4. Uses mixed precision for faster inference

## Getting Started

### 1. Start the Model Server

Start the server with:

```bash
./start_server.sh [PORT]
```

The server will:
1. Load the model once on startup (this takes ~3-4 minutes)
2. Keep the model in memory for fast inference
3. Provide an API endpoint at `/api/analyze`
4. Listen on port 5000 by default (or specify a custom port)

### 2. Use the Client With Specialized Prompts

Once the server is running, you can use the client to send inference requests with specialized medical prompts:

```bash
python client.py --img_path /path/to/image.jpg [options]
```

#### Prompt Options

The client supports specialized prompt types for different medical analysis scenarios:

```bash
# Use a specialized prompt for abnormality detection
python client.py --img_path /path/to/image.jpg --prompt_type abnormality

# Use a specialized prompt for thoracic analysis
python client.py --img_path /path/to/image.jpg --prompt_type thoracic

# Custom question
python client.py --img_path /path/to/image.jpg --question "Your custom question"
```

Available prompt types:
- `general` (default): Comprehensive analysis
- `modality`: Modality recognition
- `anatomy`: Anatomical mapping
- `abnormality`: Abnormality detection
- `congenital`: Congenital variant recognition
- `thoracic`: Thoracic analysis
- `abdominal`: Abdominal assessment
- `neuro`: Neuroimaging interpretation
- `brain_viability`: Brain viability assessment
- `msk`: Musculoskeletal examination
- `genitourinary`: Genitourinary system analysis
- `diagnosis`: Differential diagnosis

## API Usage

The server exposes a simple REST API:

### Health Check
```
GET /api/healthcheck
```

### Analyze Image
```
POST /api/analyze
Content-Type: application/json

{
  "img_path": "/path/to/image.jpg",
  "question": "Analyze this medical image..."
}
```

## Performance Comparison

| Method | First Run | Subsequent Runs |
|--------|-----------|-----------------|
| Original Script | 3-5 minutes | 3-5 minutes |
| Model Server | 3-4 minutes (load) + 30-40 seconds (inference) | 30-40 seconds |

## Troubleshooting

- **Permissions**: Make sure scripts are executable (`chmod +x *.sh`)
- **Dependencies**: The start_server script will automatically install Flask and requests if needed
- **Prompts**: Ensure `medical0.tools.txt` is present in the project root for specialized prompts
- **CUDA issues**: If you encounter CUDA errors, try reducing the server to CPU mode by setting the environment variable `CUDA_VISIBLE_DEVICES=""` 