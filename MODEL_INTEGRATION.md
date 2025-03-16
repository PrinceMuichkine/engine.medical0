# HealthGPT Fast Model Inference Integration

This document explains how the HealthGPT API has been optimized to provide fast model inference using an integrated model server.

## Key Features

- **Single-Process Architecture**: The API now loads the model once at startup and keeps it in memory, eliminating the need to run a separate model server process.
- **Significantly Faster Inference**: After the initial model loading (~3-4 minutes), subsequent inferences take only 30-40 seconds rather than 3-5 minutes.
- **Automatic Fallback**: If the optimized inference fails for any reason, the system automatically falls back to the original script-based approach.
- **Image Caching**: Processed images are cached to avoid redundant processing for repeated analyses of the same image.

## How It Works

1. A new module `engine/model_engine.py` handles model loading and inference
2. The API (`apy.py`) uses this module directly, instead of calling external scripts
3. The model is loaded once at API startup and kept in memory for all subsequent requests

## Performance Comparison

| Method | First Run | Subsequent Runs |
|--------|-----------|-----------------|
| Previous Script-Based | 3-5 minutes | 3-5 minutes |
| Integrated Model Server | 3-4 minutes (load) + 30-40 seconds (inference) | 30-40 seconds |

## Starting the API Server

The simplest way to start the server is using the provided startup script:

```bash
cd engine
./start_server.sh
```

This will start the server with the default options. You can customize the server startup with these options:

```bash
# Start with Phi-4 model instead of Phi-3
./start_server.sh --phi4

# Start on a different port
./start_server.sh --port 8080

# Start without preloading models (they'll be loaded on first request)
./start_server.sh --no-preload

# See all options
./start_server.sh --help
```

Alternatively, you can run the Python script directly:

```bash
cd engine
python apy.py --preload-models --port 5001
```

The server will:
1. Load the model once on startup (this takes ~3-4 minutes)
2. Keep the model in memory for fast inference
3. Listen on the specified port for API requests

## API Usage

The API endpoints remain the same:

### Analyze Image
```
POST /analyze-image
Content-Type: application/json

{
  "image": "base64 encoded image or URL",
  "question": "Analyze this medical image...",
  "analysis_type": "general",
  "use_phi4": false
}
```

### Reconstruct Image
```
POST /reconstruct-image
Content-Type: application/json

{
  "image": "base64 encoded image or URL",
  "question": "Enhance this medical image...",
  "analysis_type": "general",
  "use_phi4": false
}
```

## Troubleshooting

- **Memory Issues**: If you encounter CUDA out of memory errors, try restarting the server.
- **Model Loading Failures**: Check that the weight files are correctly located in one of these directories:
  - `./weights/`
  - `../weights/`
  - `../../weights/`
- **Fallback Operation**: If you notice that inference is taking much longer than 30-40 seconds, the server might have fallen back to script-based inference. Check the logs for errors.

## Technical Details

The system integrates the following components:

- **Flask API Server** (`apy.py`): Handles HTTP requests and responses
- **Model Engine** (`model_engine.py`): Loads and manages the AI model in memory
- **LLaVA Framework**: Provides the underlying vision-language models (Phi-3 and Phi-4)

All the code for directly working with the model is abstracted into the `model_engine.py` module, making the main API code cleaner and easier to maintain. 