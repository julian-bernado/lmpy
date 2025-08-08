#!/usr/bin/env python3
"""
Test single answer functionality with system prompts.
"""

from lmpy import Client
from lmpy.server.manager import LlamaServer
from test_utils import get_available_models, get_primary_model


def test_single_answer(model_path, model_name, verbose=False):
    """Test getting a single answer with system prompt."""
    print(f"\n=== Testing Single Answer with {model_name} ===")
    
    try:
        # Start server automatically
        with LlamaServer(
            model=model_path,
            port=8080,
            ctx_size=4096,
            n_gpu_layers=-1,  # Use all GPU layers if available
            alias=model_name,
            verbose=verbose  # Control server output verbosity
        ) as server:
            if verbose:
                print(f"🚀 Started {model_name} server at {server.base_url}")
            else:
                print(f"🚀 Started {model_name} server (quiet mode)")
            
            # Create client connected to our server
            llm = Client(
                base_url=server.base_url,
                system_prompt="You are a helpful assistant.",
                timeout=30.0
            )
            
            # Test simple question
            prompt = "What is the capital of France?"
            print(f"Prompt: {prompt}")
            
            answer = llm.answer(prompt, max_tokens=1024, temperature=0.7)
            print(f"Answer: {answer}")
            
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        return False
    
    return True


def main():
    """Run single answer test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test single answer functionality")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show verbose server output")
    args = parser.parse_args()
    
    print("🚀 Testing single answer functionality...")
    print("This will automatically start and stop a llama-server process.")
    if not args.verbose:
        print("ℹ️  Use --verbose to see server startup logs")
    print()
    
    # Find available models
    print("🔍 Scanning for available models...")
    models = get_available_models(verbose=args.verbose)
    print(f"\n✓ Found {len(models)} available models")
    
    # Use the first available model
    model_path = "/Users/jubernado/repos/lmpy/models/openai/gpt-oss-20b-q4.gguf"
    model_name = "gpt-oss-20b"
    model_type = "harmony"
    print(f"\n🎯 Running test with: {model_name} ({model_type})")
    
    # Run test
    success = test_single_answer(model_path, model_name, verbose=args.verbose)
    
    if success:
        print(f"\n✅ Single answer test completed successfully!")
    else:
        print(f"\n❌ Single answer test failed!")
    
    print("Server process has been automatically stopped.")


if __name__ == "__main__":
    main()
