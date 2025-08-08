#!/usr/bin/env python3
"""
Test message conversation functionality - multi-turn conversations.
"""

from lmpy import Client
from lmpy.server.manager import LlamaServer
from test_utils import get_available_models, get_primary_model


def test_message_conversations(model_path, model_name, model_type, verbose=False):
    """Test using message lists for multi-turn conversations."""
    print(f"\n=== Testing Message Conversations with {model_name} ===")
    
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
                system_prompt="You are a helpful coding assistant.",
                timeout=30.0
            )
            
            # Create a multi-turn conversation
            messages = [
                {"role": "user", "content": "What is a Python list?"},
                {"role": "assistant", "content": "A Python list is a mutable, ordered collection of items that can store multiple values in a single variable."},
                {"role": "user", "content": "How do I add an item to it?"}
            ]
            
            print("Multi-turn conversation:")
            for msg in messages:
                print(f"  {msg['role'].title()}: {msg['content']}")
            
            answer = llm.answer(messages, max_tokens=150, temperature=0.5)
            print(f"  Assistant: {answer}")
            
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        return False
    
    return True


def main():
    """Run message conversations test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test message conversations functionality")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show verbose server output")
    args = parser.parse_args()
    
    """Run message conversations test."""
    print("🚀 Testing message conversation functionality...")
    print("This demonstrates multi-turn conversations using message lists.")
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
    success = test_message_conversations(model_path, model_name, model_type, verbose=args.verbose)
    
    if success:
        print(f"\n✅ Message conversations test completed successfully!")
    else:
        print(f"\n❌ Message conversations test failed!")
    
    print("Server process has been automatically stopped.")


if __name__ == "__main__":
    main()
