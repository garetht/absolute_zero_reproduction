#!/usr/bin/env python3
"""
Test runner for compute module tests (advantages and rewards).

Usage:
    python tests/run_compute_tests.py              # Run all compute tests
    python tests/run_compute_tests.py -v           # Run with verbose output
    python tests/run_compute_tests.py advantages   # Run only advantages tests
    python tests/run_compute_tests.py rewards      # Run only rewards tests
    python tests/run_compute_tests.py -k "test_name"  # Run specific test by name
"""

import sys
import subprocess
import os


def main():
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Check for specific test file requests
    if len(sys.argv) > 1:
        if sys.argv[1] == "advantages":
            cmd.append("tests/test_advantages.py")
        elif sys.argv[1] == "rewards":
            cmd.append("tests/test_rewards.py")
        elif sys.argv[1] in ["-v", "--verbose"]:
            cmd.append("-v")
            cmd.extend(["tests/test_advantages.py", "tests/test_rewards.py"])
        elif sys.argv[1] in ["-k"]:
            # Pass through -k flag for test filtering
            cmd.extend(sys.argv[1:])
            cmd.extend(["tests/test_advantages.py", "tests/test_rewards.py"])
        else:
            # Pass all arguments to pytest
            cmd.extend(sys.argv[1:])
    else:
        # Default: run both test files
        cmd.extend(["tests/test_advantages.py", "tests/test_rewards.py"])
    
    # Add common pytest options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--color=yes", # Colored output
    ])
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the tests
    result = subprocess.run(cmd)
    
    # Exit with same code as pytest
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
