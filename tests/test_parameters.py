#!/usr/bin/env python3
"""
Test parameter variations - temperature, top_p, etc.
"""

from lmpy import Client
from lmpy.server.manager import LlamaServer
from test_utils import get_available_models, get_primary_model


def test_parameter_variations(model_path, model_name, model_type, verbose=False):
    """Test different parameter settings."""
    print(f"\n=== Testing Parameter Variations with {model_name} ===")
    
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
                system_prompt="You are a creative writer."
            )
            prompt = "Write a one-sentence description of a magical forest."
            
            # Test different temperatures
            temperatures = [0.1, 0.5, 0.9]
            
            for temp in temperatures:
                print(f"\nTemperature: {temp}")
                answer = llm.answer(
                    prompt, 
                    max_tokens=100, 
                    temperature=temp,
                    top_p=0.95
                )
                print(f"Answer: {answer}")
                
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        return False
    
    return True


def main():
    """Run parameter variations test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test parameter variations functionality")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show verbose server output")
    args = parser.parse_args()
    
    """Run parameter variations test."""
    print("🚀 Testing parameter variations...")
    print("This tests how different temperature settings affect creativity.")
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
    success = test_parameter_variations(model_path, model_name, model_type, verbose=args.verbose)
    
    if success:
        print(f"\n✅ Parameter variations test completed successfully!")
    else:
        print(f"\n❌ Parameter variations test failed!")
    
    print("Server process has been automatically stopped.")


if __name__ == "__main__":
    main()
