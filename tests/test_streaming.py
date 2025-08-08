#!/usr/bin/env python3
"""
Test streaming response functionality for real-time output.
"""

from lmpy import Client
from lmpy.server.manager import LlamaServer
from test_utils import get_available_models, get_primary_model
import time


def test_streaming_response(model_path, model_name, model_type, verbose=False):
    """Test streaming responses for real-time output."""
    print(f"\n=== Testing Streaming Response with {model_name} ===")
    
    try:
        with LlamaServer(
            model=model_path,
            port=8080,
            ctx_size=4096,
            n_gpu_layers=-1,
            alias=model_name
            verbose=verbose  # Control server output verbosity
        ) as server:
            if verbose:
                print(f"🚀 Started {model_name} server at {server.base_url}")
            else:
                print(f"🚀 Started {model_name} server (quiet mode)")
            
            llm = Client(
                base_url=server.base_url,
                system_prompt="You are a storyteller. Tell engaging short stories.",
                timeout=30.0
            )
            
            prompt = "Tell me a very short story about a robot learning to paint."
            print(f"Prompt: {prompt}")
            print("Streaming response:")
            print("-" * 50)
            
            # Stream the response token by token
            full_response = ""
            for chunk in llm.answer_stream(prompt, max_tokens=200, temperature=0.8):
                print(chunk, end="", flush=True)
                full_response += chunk
                time.sleep(0.02)  # Small delay to see streaming effect
            
            print("\n" + "-" * 50)
            print(f"✓ Complete response length: {len(full_response)} characters")
            
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        return False
    
    return True


def main():
    """Run streaming response test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test streaming response functionality")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show verbose server output")
    args = parser.parse_args()
    
    """Run streaming response test."""
    print("🚀 Testing streaming response functionality...")
    print("This will display tokens in real-time as they're generated.")
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
    success = test_streaming_response(model_path, model_name, model_type, verbose=args.verbose)
    
    if success:
        print(f"\n✅ Streaming response test completed successfully!")
    else:
        print(f"\n❌ Streaming response test failed!")
    
    print("Server process has been automatically stopped.")


if __name__ == "__main__":
    main()
