#!/usr/bin/env python3
"""
Test grammar-constrained generation.
"""

from lmpy import Client
from lmpy.server.manager import LlamaServer
from test_utils import get_available_models, get_primary_model


def test_grammar_constraints(model_path, model_name, model_type, verbose=False):
    """Test grammar-constrained generation."""
    print(f"\n=== Testing Grammar Constraints with {model_name} ===")
    
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
            
            llm = Client(
                base_url=server.base_url,
                system_prompt="You are a helpful assistant.",
                timeout=60.0
            )
            
            prompt = "Which of the numbers from 1 to 30 are prime? Respond in a bracketed list like [num1, num2, num3, ...]"
            
            # This test requires a grammar file and llama.cpp compiled with grammar support
            grammar_file = "grammars/numerical_list.gbnf"  # Assuming this exists
            
            answer = llm.answer(
                prompt,
                max_tokens=1024,
                temperature=0.5,
                grammar=grammar_file
            )
            print(f"Prompt: {prompt}")
            print(f"Grammar-constrained answer: {answer}")
            
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        return False
    
    return True


def main():
    """Run grammar constraints test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test grammar constraints functionality")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show verbose server output")
    args = parser.parse_args()
    
    """Run grammar constraints test."""
    print("🚀 Testing grammar-constrained generation...")
    print("This tests structured output generation using grammar files.")
    print("Note: Requires grammar files in ./grammars/ and llama.cpp with grammar support.")
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
    # model_path = "/Users/jubernado/repos/lmpy/models/qwen/qwen3-14b-q4.gguf"
    # model_name = "qwen3-14b"
    # model_type = "standard"    
    print(f"\n🎯 Running test with: {model_name} ({model_type})")
    
    # Run test
    success = test_grammar_constraints(model_path, model_name, model_type, verbose=args.verbose)
    
    if success:
        print(f"\n✅ Grammar constraints test completed successfully!")
    else:
        print(f"\n❌ Grammar constraints test failed!")
    
    print("Server process has been automatically stopped.")


if __name__ == "__main__":
    main()
