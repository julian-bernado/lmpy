#!/usr/bin/env python3
"""
Test multi-answer functionality - processing multiple prompts sequentially.
"""

from lmpy import Client
from lmpy.server.manager import LlamaServer
from test_utils import get_available_models, get_primary_model


def test_multi_answers(model_path, model_name, model_type, verbose=False):
    """Test getting multiple answers from a list of prompts."""
    print(f"\n=== Testing Multi Answers with {model_name} ===")
    
    try:
        with LlamaServer(
            model=model_path,
            port=8080,
            ctx_size=4096,
            n_gpu_layers=-1,
            alias=model_name,
            verbose=verbose
        ) as server:
            if verbose:
                print(f"🚀 Started {model_name} server at {server.base_url}")
            else:
                print(f"🚀 Started {model_name} server (quiet mode)")
            
            # Create client
            llm = Client(
                base_url=server.base_url,
                system_prompt="You are a helpful assistant that provides clear, factual answers.",
                timeout=60.0
            )
            
            # List of prompts to answer
            prompts = [
                "What is 2 + 2?",
                "Name one primary color.",
                "What is the largest planet in our solar system?",
                "Implement the fibonacci sequence in python as an iterator."
            ]
            
            print("Processing multiple prompts...")
            # Get all answers with progress bar
            answers = llm.multi_answer(
                prompts, 
                progress=True,  # Show progress bar
                show=True,      # Print each prompt/answer pair
                max_tokens=1024,
                temperature=0.3
            )
            
            print(f"\n✓ Received {len(answers)} answers total.")
            
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        return False
    
    return True


def main():
    """Run multi-answer test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test multi-answer functionality")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show verbose server output")
    args = parser.parse_args()
    
    """Run multi-answer test."""
    print("🚀 Testing multi-answer functionality...")
    print("This will process multiple prompts sequentially with a progress bar.")
    if not args.verbose:
        print("ℹ️  Use --verbose to see server startup logs")
    print()
    
    # Find available models
    print("🔍 Scanning for available models...")
    models = get_available_models(verbose=args.verbose)
    print(f"\n✓ Found {len(models)} available models")
    
    # Use the first available model
    # model_path = "/Users/jubernado/repos/lmpy/models/openai/gpt-oss-20b-q4.gguf"
    # model_name = "gpt-oss-20b"
    # model_type = "harmony"    
    model_path = "/Users/jubernado/repos/lmpy/models/qwen/qwen3-14b-q4.gguf"
    model_name = "qwen3-14b"
    model_type = "standard"    

    print(f"\n🎯 Running test with: {model_name} ({model_type})")
    
    # Run test
    success = test_multi_answers(model_path, model_name, model_type, verbose=args.verbose)
    
    if success:
        print(f"\n✅ Multi-answer test completed successfully!")
    else:
        print(f"\n❌ Multi-answer test failed!")
    
    print("Server process has been automatically stopped.")


if __name__ == "__main__":
    main()
