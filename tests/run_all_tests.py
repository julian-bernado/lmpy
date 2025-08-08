#!/usr/bin/env python3
"""
Run all lmpy tests.

This script runs all available test files and reports results.
"""

import subprocess
import sys
from pathlib import Path


def get_test_files():
    """Get all test files in the current directory."""
    test_dir = Path(__file__).parent
    test_files = []
    
    # Basic functionality tests
    basic_tests = [
        "test_single_answer.py",
        "test_system_prompts.py", 
        "test_multi_answers.py",
        "test_harmony_vs_standard.py"
    ]
    
    # Advanced functionality tests
    advanced_tests = [
        "test_streaming.py",
        "test_conversations.py",
        "test_parameters.py",
        "test_tokens.py",
        "test_grammar.py"
    ]
    
    all_tests = basic_tests + advanced_tests
    
    for test_file in all_tests:
        test_path = test_dir / test_file
        if test_path.exists():
            test_files.append((test_file, "Basic" if test_file in basic_tests else "Advanced"))
    
    return test_files


def run_test(test_file, verbose=False):
    """Run a single test file and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)
    
    try:
        cmd = [sys.executable, test_file]
        if verbose:
            cmd.append("--verbose")
            
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=Path(__file__).parent
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run {test_file}: {e}")
        return False


def main():
    """Run all tests and report results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all lmpy tests")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show verbose server output for all tests")
    args = parser.parse_args()
    
    print("🚀 Running all lmpy tests...")
    print("This will run each test file individually with automatic server management.")
    
    if not args.verbose:
        print("ℹ️  Use --verbose to see server startup logs")
    print()
    
    test_files = get_test_files()
    
    if not test_files:
        print("❌ No test files found!")
        sys.exit(1)
    
    print(f"Found {len(test_files)} test files:")
    for test_file, category in test_files:
        print(f"  - {test_file} ({category})")
    
    print("\nStarting test execution...")
    
    results = {}
    basic_passed = 0
    advanced_passed = 0
    basic_total = 0
    advanced_total = 0
    
    for test_file, category in test_files:
        success = run_test(test_file, verbose=args.verbose)
        results[test_file] = success
        
        if category == "Basic":
            basic_total += 1
            if success:
                basic_passed += 1
        else:
            advanced_total += 1
            if success:
                advanced_passed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    print(f"\nBasic Tests: {basic_passed}/{basic_total} passed")
    print(f"Advanced Tests: {advanced_passed}/{advanced_total} passed")
    print(f"Overall: {basic_passed + advanced_passed}/{basic_total + advanced_total} passed")
    
    print("\nDetailed Results:")
    for test_file, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_file}: {status}")
    
    if basic_passed + advanced_passed == basic_total + advanced_total:
        print(f"\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
