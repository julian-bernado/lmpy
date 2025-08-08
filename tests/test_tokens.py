#!/usr/bin/env python3
"""
Test token counting functionality.
"""

from lmpy import Client
from lmpy.server.manager import LlamaServer
from test_utils import get_available_models, get_primary_model


def test_token_counting(model_path, model_name, model_type, verbose=False):
    """Test token counting functionality."""
    print(f"\n=== Testing Token Counting with {model_name} ===")
    
    try:
        with LlamaServer(
            model=model_path,
            port=8080,
            ctx_size=4096,
            n_gpu_layers=-1,
            alias=model_name,
            verbose=verbose  # Control server output verbosity
        ) as server:
            if verbose:
                print(f"🚀 Started {model_name} server at {server.base_url}")
            else:
                print(f"🚀 Started {model_name} server (quiet mode)")
            
            llm = Client(base_url=server.base_url)
            
            texts = [
                "Hello world!",
                "This is a longer sentence with more words to count.",
                "The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing."
            ]
            
            for text in texts:
                token_count = llm.num_tokens(text)
                print(f"Text: '{text}'")
                print(f"Token count: {token_count}")
                print(f"Characters: {len(text)}")
                print(f"Ratio: {len(text)/token_count:.2f} chars/token\n")
                
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        return False
    
    return True


def main():
    """Run token counting test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test token counting functionality")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show verbose server output")
    args = parser.parse_args()
    
    """Run token counting test."""
    print("🚀 Testing token counting functionality...")
    print("This shows how text is tokenized by the model.")
    if not args.verbose:
        print("ℹ️  Use --verbose to see server startup logs")
    print()
    
    # Find available models
    print("🔍 Scanning for available models...")
    models = get_available_models(verbose=args.verbose)
    print(f"\n✓ Found {len(models)} available models")
    
    # Use the first available model
    model_path, model_name, model_type = get_primary_model(models)
    print(f"\n🎯 Running test with: {model_name} ({model_type})")
    
    # Run test
    success = test_token_counting(model_path, model_name, model_type, verbose=args.verbose)
    
    if success:
        print(f"\n✅ Token counting test completed successfully!")
    else:
        print(f"\n❌ Token counting test failed!")
    
    print("Server process has been automatically stopped.")


if __name__ == "__main__":
    main()
