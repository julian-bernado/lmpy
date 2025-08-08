#!/usr/bin/env python3
"""
Test system prompt variations with the same question.
"""

from lmpy import Client
from lmpy.server.manager import LlamaServer
from test_utils import get_available_models, get_primary_model


def test_system_prompt_variations(model_path, model_name, model_type, verbose=False):
    """Test different system prompts with the same question."""
    print(f"\n=== Testing System Prompt Variations with {model_name} ===")
    
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
            
            question = "Explain machine learning in simple terms."
            
            # Test 1: Concise assistant
            llm1 = Client(
                base_url=server.base_url,
                system_prompt="You are a concise assistant. Give very brief answers."
            )
            print("System: Concise assistant")
            print(f"Question: {question}")
            answer1 = llm1.answer(question, max_tokens=1024, temperature=0.5)
            print(f"Answer: {answer1}\n")
            
            # Test 2: Detailed teacher
            llm2 = Client(
                base_url=server.base_url,
                system_prompt="You speak in Shakespearean tongue."
            )
            print("System: Patient teacher")
            print(f"Question: {question}")
            answer2 = llm2.answer(question, max_tokens=1024, temperature=0.5)
            print(f"Answer: {answer2}")
            
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        return False
    
    return True


def main():
    """Run system prompt variations test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test system prompt variations functionality")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show verbose server output")
    args = parser.parse_args()
    
    """Run system prompt variations test."""
    print("🚀 Testing system prompt variations...")
    print("This will test how different system prompts affect the same question.")
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
    success = test_system_prompt_variations(model_path, model_name, model_type, verbose=args.verbose)
    
    if success:
        print(f"\n✅ System prompt variations test completed successfully!")
    else:
        print(f"\n❌ System prompt variations test failed!")
    
    print("Server process has been automatically stopped.")


if __name__ == "__main__":
    main()
