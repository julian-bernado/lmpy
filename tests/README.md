# lmpy Tests

This directory contains individual test files that demonstrate and validate the functionality of the lmpy library.

## Overview

All tests automatically manage llama-server processes - no manual setup required! Just run any test file and it will:

1. 🔍 Scan for available models in `./models/`
2. 🚀 Start the appropriate llama-server process
3. 🧪 Run the test functionality 
4. 🛑 Automatically stop the server when done

## Test Files

### Basic Functionality
- **`test_single_answer.py`** - Simple Q&A with system prompts
- **`test_system_prompts.py`** - How different system prompts affect responses  
- **`test_multi_answers.py`** - Processing multiple prompts sequentially
- **`test_harmony_vs_standard.py`** - Comparing Harmony vs standard models

### Advanced Features
- **`test_streaming.py`** - Real-time token streaming
- **`test_conversations.py`** - Multi-turn conversations
- **`test_parameters.py`** - Temperature and parameter variations
- **`test_tokens.py`** - Token counting functionality
- **`test_grammar.py`** - Grammar-constrained generation

### Utilities
- **`test_utils.py`** - Shared utilities for model discovery
- **`run_all_tests.py`** - Run all tests and report results

## Usage

### Run Individual Tests
```bash
cd tests
uv run test_single_answer.py
# etc.
```

### Run All Tests
```bash
cd tests
uv run run_all_tests.py
```

## Requirements

### Models
Place GGUF model files in the `./models/` directory following this structure:
```
models/
├── qwen/
│   ├── qwen3-8b-q4.gguf
│   ├── qwen3-14b-q4.gguf
│   └── qwen3-32b-q4.gguf
├── openai/
│   └── gpt-oss-20b-q4.gguf
├── google/
│   ├── gemma-3-4b-q4.gguf
│   └── gemma-3-12b-q4.gguf
└── allenai/
    └── olmo-2-1124-13b-q4.gguf
```

### llama-server Binary
Ensure `llama-server` is available in your PATH or set the `LMPY_LLAMA_SERVER_BIN` environment variable.

## Model Types

### Standard Models
- **qwen3**, **gemma-3**, **olmo-2** - Traditional chat models
- Use `system_prompt` for instructions
- Return simple string responses

### Harmony Models  
- **gpt-oss** - Advanced reasoning models
- Support `developer_instructions` for fine-grained control
- Can return `HarmonyResponse` objects with thinking process
- Configure with `reasoning_effort` parameter

## Notes

- Tests automatically detect model types and adjust behavior accordingly
- Each test runs in isolation with its own server instance
- Servers use optimal settings (GPU layers, context size, etc.)
- All cleanup is handled automatically via context managers
