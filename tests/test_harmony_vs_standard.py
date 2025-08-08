#!/usr/bin/env python3
"""
Test differences between Harmony and standard models.
"""

from lmpy import Client
from lmpy.server.manager import LlamaServer
from test_utils import get_available_models


def test_harmony_vs_standard(models, verbose: bool = False):
    """Test the differences between Harmony and standard models."""
    print(f"\n=== Testing Harmony vs Standard Models ===")
    
    # Find a standard model and a harmony model if available
    standard_model = next((m for m in models if m[2] == "standard"), None)
    harmony_model = next((m for m in models if m[2] == "harmony"), None)
    
    prompt = "Write a short poem about coding."
    
    if standard_model:
        model_path, model_name, model_type = standard_model
        print(f"\n--- Testing Standard Model: {model_name} ---")
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
                    system_prompt="You are a creative poet who writes technical poems."
                )
                
                answer = llm.answer(prompt, max_tokens=200, temperature=0.8)
                print(f"Standard Model Answer: {answer}")
                
        except Exception as e:
            print(f"❌ Error with standard model {model_name}: {e}")
    
    if harmony_model:
        model_path, model_name, model_type = harmony_model
        print(f"\n--- Testing Harmony Model: {model_name} ---")
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
                
                llm_harmony = Client(
                    base_url=server.base_url,
                    system_prompt="You are a helpful assistant.",  # Fallback for standard models
                    developer_instructions="You are a creative poet who writes technical poems.",  # Harmony-specific
                    reasoning_effort="medium"
                )
                
                answer = llm_harmony.answer(prompt, max_tokens=200, temperature=0.8)
                print(f"Harmony Model Answer: {answer}")
                
        except Exception as e:
            print(f"❌ Error with harmony model {model_name}: {e}")
    
    if not standard_model and not harmony_model:
        print("❌ No suitable models found for this test")
        return False
    
    return True


def main():
    """Run harmony vs standard test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test harmony vs standard functionality")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show verbose server output")
    args = parser.parse_args()
    
    """Run Harmony vs Standard models test."""
    print("🚀 Testing Harmony vs Standard models...")
    print("This compares how different model types handle the same prompt.")
    if not args.verbose:
        print("ℹ️  Use --verbose to see server startup logs")
    print()
    
    # Find available models
    print("🔍 Scanning for available models...")
    models = get_available_models(verbose=args.verbose)
    print(f"\n✓ Found {len(models)} available models")
    
    # Run test
    success = test_harmony_vs_standard(models)
    
    if success:
        print(f"\n✅ Harmony vs Standard test completed successfully!")
    else:
        print(f"\n❌ Harmony vs Standard test failed!")
    
    print("All server processes have been automatically stopped.")


if __name__ == "__main__":
    main()
