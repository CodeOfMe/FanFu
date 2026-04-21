#!/usr/bin/env python3
"""Test FanFu package installation and basic functionality."""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*50}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*50}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {description} passed")
        if result.stdout:
            print(result.stdout[:500])
        return True
    else:
        print(f"❌ {description} failed")
        if result.stderr:
            print(result.stderr[:500])
        return False


def main():
    """Run package tests."""
    print("FanFu Package Test")
    print("=" * 50)

    tests = [
        ("python -c 'import fanfu; print(fanfu.__version__)'", "Package import"),
        ("python -c 'from fanfu import convert_gguf_to_hf; print(\"OK\")'", "API import"),
        ("python -c 'from fanfu import compare_weights; print(\"OK\")'", "Compare import"),
        ("python -c 'from fanfu import ToolResult; print(\"OK\")'", "ToolResult import"),
        ("fanfu --version", "CLI version"),
        ("fanfu --help", "CLI help"),
        ("python -m pytest tests/test_core.py -v --tb=short", "Core tests"),
    ]

    passed = 0
    failed = 0

    for cmd, desc in tests:
        if run_command(cmd, desc):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Package Test Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
