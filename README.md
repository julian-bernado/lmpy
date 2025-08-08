# lmpy

A Python client for llama.cpp HTTP server with OpenAI Harmony support.

## Features

- **Simple API**: Pass a string prompt, get a string answer
- **Automatic server management**: No manual setup required
- **Model type detection**: Automatically detects standard vs. Harmony models
- **Streaming support**: Real-time token streaming
- **Multi-model support**: Run different models on different ports
- **Token counting**: Built-in tokenization utilities
- **Embedding support**: Generate embeddings with compatible models

## Installation

Since this is a local development project (not yet published on PyPI), you have several options:

### Option 1: Install in Development Mode (Recommended)
```bash
# Clone/download this repository
git clone <repository-url>
cd lmpy

# Install in editable mode with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Option 2: Install from Local Path
```bash
# From another project directory
uv pip install /path/to/lmpy
# Or with pip
pip install /path/to/lmpy
```

### Option 3: Add to Python Path
```python
# In your script, add the lmpy src directory to Python path
import sys
sys.path.insert(0, '/path/to/lmpy/src')
from lmpy import Client
```

### Option 4: Use uv for Local Development
```bash
# In your project directory, add as a local dependency
uv add --editable /path/to/lmpy
```

## Quick Start

### Prerequisites
1. **llama-server binary**: Download from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) or build from source
2. **GGUF models**: Download compatible models and place in `./models/` directory
3. **Python 3.12+**: Required for this project

### Setting Up the Development Environment
```bash
# Clone the repository
git clone <repository-url>
cd lmpy

# Install dependencies
uv sync
# Or with pip: pip install -e .[dev]

# Set up model directory and download some models
mkdir -p models/qwen
# Place your .gguf files in the appropriate subdirectories
```

### Running the Examples
```bash
# Test the installation by running example tests
cd tests
uv run test_single_answer.py

# Or run all tests to see everything working
uv run run_all_tests.py
```

### Using from Another Project

If you want to use lmpy from another local repository:

```python
# Method 1: Install lmpy in your project's environment
# (from your project directory)
# uv add --editable /path/to/lmpy

# Method 2: Add to Python path (quick & dirty)
import sys
sys.path.append('/path/to/lmpy/src')

from lmpy import Client
from lmpy.server.manager import LlamaServer
from lmpy.paths import find

# Make sure to set the model directory if lmpy is in a different location
import os
os.environ['LMPY_MODEL_DIR'] = '/path/to/lmpy/models'

# Now use normally
model_path = find("qwen3-8b-q4")
with LlamaServer(model=model_path, port=8080, n_gpu_layers=-1) as server:
    llm = Client(base_url=server.base_url)
    answer = llm.answer("Hello!")
    print(answer)
```

### 1. Load a Model and Get Started

```python
from lmpy import Client
from lmpy.server.manager import LlamaServer
from lmpy.paths import find

# Find a model file (looks in ./models/ by default)
model_path = find("qwen3-8b-q4")  # Case-insensitive search

# Start server and create client
with LlamaServer(
    model=model_path,
    port=8080,
    ctx_size=4096,
    n_gpu_layers=-1  # Use all GPU layers
) as server:
    llm = Client(base_url=server.base_url)
    
    # Get a response
    answer = llm.answer("What is the capital of France?")
    print(answer)
```

### 2. Basic Usage Patterns

#### Simple Q&A
```python
from lmpy import Client

# If you already have a running llama-server
llm = Client()  # Connects to http://127.0.0.1:8080 by default

# Ask a question
response = llm.answer("Explain quantum computing in simple terms.")
print(response)
```

#### Set System Prompt
```python
llm = Client(
    system_prompt="You are a helpful coding assistant specialized in Python."
)

answer = llm.answer("How do I create a list comprehension?")
print(answer)

# Or change it later
llm.set_system_prompt("You are a creative writer.")
```

#### Get Multiple Responses
```python
questions = [
    "What is 2 + 2?",
    "Name the largest planet.",
    "What is photosynthesis?"
]

# Process all questions with progress bar
answers = llm.multi_answer(
    questions,
    progress=True,  # Show progress bar
    show=True,      # Print each Q&A pair
    temperature=0.7
)

# Access individual answers
for i, answer in enumerate(answers):
    print(f"Q{i+1}: {questions[i]}")
    print(f"A{i+1}: {answer}\n")
```

#### Stream Responses
```python
# Get tokens as they arrive
print("Response: ", end="")
for token in llm.answer_stream("Tell me a short story about AI."):
    print(token, end="", flush=True)
print()  # New line when done
```

#### Check Token Count
```python
text = "This is a sample text to count tokens."
token_count = llm.num_tokens(text)
print(f"Text: '{text}'")
print(f"Tokens: {token_count}")
print(f"Ratio: {len(text)/token_count:.2f} chars/token")
```

### 3. Advanced Configuration

#### Server Parameters
```python
with LlamaServer(
    model=model_path,
    port=8080,
    ctx_size=8192,      # Context window size
    batch=512,          # Batch size
    n_gpu_layers=-1,    # GPU layers (-1 = all)
    alias="my-model",   # Server alias
    verbose=True        # Show server logs
) as server:
    # Your code here
```

#### Client Parameters
```python
llm = Client(
    base_url="http://localhost:8080",
    timeout=30.0,
    system_prompt="You are a helpful assistant.",
    # Harmony model options:
    reasoning_effort="medium",  # high/medium/low
    developer_instructions="Be concise and accurate.",
    enable_builtin_python=False,
    enable_builtin_browser=False
)
```

#### Generation Parameters
```python
answer = llm.answer(
    "Write a haiku about programming.",
    max_tokens=100,
    temperature=0.8,      # Creativity (0.0-2.0)
    top_p=0.95,          # Nucleus sampling
    presence_penalty=0.1, # Avoid repetition
    frequency_penalty=0.1,
    stop=["END", "\n\n"]  # Stop sequences
)
```

### 4. Model Types

#### Standard Models (qwen3, gemma-3, olmo-2)
```python
# Use system_prompt for instructions
llm = Client(system_prompt="You are a helpful assistant.")
answer = llm.answer("Hello!")  # Returns simple string
```

#### Harmony Models (gpt-oss)
```python
# Support advanced reasoning with developer instructions
llm = Client(
    reasoning_effort="high",
    developer_instructions="Think step by step and show your reasoning."
)
answer = llm.answer("Solve this logic puzzle: ...")
# Returns clean final answer (reasoning is processed internally)
```

### 5. Model Management

#### Find Models
```python
from lmpy.paths import find, list_gguf

# Find specific model (case-insensitive)
model_path = find("qwen3-8b")  # Finds qwen3-8b-q4.gguf

# List all available models
all_models = list_gguf()
for model in all_models:
    print(model)

# Set custom model directory
model_path = find("my-model", model_dir="/path/to/models")
```

#### Recommended Directory Structure
```
models/
├── qwen/
│   ├── qwen3-8b-q4.gguf
│   ├── qwen3-14b-q4.gguf
│   └── qwen3-embedding-0.6b-q8.gguf
├── google/
│   ├── gemma-3-4b-q4.gguf
│   └── gemma-3-12b-q4.gguf
├── openai/
│   └── gpt-oss-20b-q4.gguf
└── allenai/
    └── olmo-2-1124-13b-q4.gguf
```

### 6. Embeddings

```python
# Start embedding model server
embedding_model = find("qwen3-embedding-0.6b")
with LlamaServer(
    model=embedding_model,
    port=8081,
    embedding=True,
    pooling="last"
) as server:
    client = Client(base_url=server.base_url)
    
    # Single embedding
    vector = client.embed("Hello world")
    print(f"Embedding dimension: {len(vector)}")
    
    # Batch embeddings
    vectors = client.embed(["Hello", "World", "AI"])
    for i, vec in enumerate(vectors):
        print(f"Text {i}: {len(vec)} dimensions")
```

### 7. Multiple Models

```python
# Run different models on different ports
models = [
    (find("qwen3-8b-q4"), 8080, "qwen-chat"),
    (find("qwen3-embedding-0.6b"), 8081, "qwen-embed")
]

# Start all servers
servers = []
for model_path, port, alias in models:
    server = LlamaServer(
        model=model_path,
        port=port,
        alias=alias,
        n_gpu_layers=-1
    )
    server.__enter__()
    servers.append(server)

try:
    # Use chat model
    chat_client = Client(base_url="http://127.0.0.1:8080")
    response = chat_client.answer("Hello!")
    
    # Use embedding model
    embed_client = Client(base_url="http://127.0.0.1:8081")
    embedding = embed_client.embed("Hello!")
    
finally:
    # Clean up all servers
    for server in servers:
        server.__exit__(None, None, None)
```

## Environment Variables

- `LMPY_BASE_URL`: Default server URL (default: `http://127.0.0.1:8080`)
- `LMPY_TIMEOUT`: Request timeout in seconds (default: `60`)
- `LMPY_MODEL_DIR`: Model directory path (default: `./models`)
- `LMPY_LLAMA_SERVER_BIN`: Path to llama-server binary

## Requirements

### llama-server Binary
Ensure `llama-server` is available in your PATH or set `LMPY_LLAMA_SERVER_BIN`:

```bash
# Download from llama.cpp releases or build from source
export LMPY_LLAMA_SERVER_BIN="/path/to/llama-server"
```

### GGUF Models
Download GGUF format models and place them in the `./models/` directory. Popular sources:
- [Hugging Face](https://huggingface.co/models?library=gguf)
- [llama.cpp model releases](https://github.com/ggerganov/llama.cpp/releases)

## Error Handling

```python
from lmpy.backends.llamacpp_http import LmpyHTTPError, LmpyTimeoutError

try:
    answer = llm.answer("Your question here")
except LmpyTimeoutError:
    print("Request timed out")
except LmpyHTTPError as e:
    print(f"HTTP error {e.status_code}: {e.message}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Tips

1. **GPU Usage**: Set `n_gpu_layers=-1` to use all available GPU layers for faster inference
2. **Context Size**: Increase `ctx_size` for longer conversations (uses more memory)
3. **Temperature**: Use 0.0 for deterministic outputs, 0.7-1.0 for creative responses
4. **Batch Processing**: Use `multi_answer()` for processing many prompts efficiently
5. **Streaming**: Use `answer_stream()` for real-time applications
6. **Model Selection**: Smaller models (4B-8B) are faster, larger models (13B+) are more capable
