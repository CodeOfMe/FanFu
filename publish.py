#!/usr/bin/env python3
"""Publish FanFu to PyPI."""

import subprocess
import sys
import re
from pathlib import Path


def bump_version():
    """Bump patch version."""
    version_file = Path("fanfu/__init__.py")
    content = version_file.read_text(encoding="utf-8")
    match = re.search(r'(__version__\s*=\s*"(\d+\.\d+\.)(\d+)")', content)
    if not match:
        print("ERROR: cannot parse version")
        sys.exit(1)

    old_v = match.group(2) + match.group(3)
    new_v = match.group(2) + str(int(match.group(3)) + 1)
    new_content = content.replace(match.group(1), f'__version__ = "{new_v}"')
    version_file.write_text(new_content, encoding="utf-8")
    print(f"  Version: {old_v} -> {new_v}")
    return new_v


def clean_builds():
    """Clean old build artifacts."""
    import shutil
    for d in ["dist", "build", "fanfu.egg-info"]:
        if Path(d).exists():
            shutil.rmtree(d)
    print("  Cleaned old builds")


def install_tools():
    """Install build tools."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "build", "twine", "-q"])
    print("  Build tools installed")


def build_package():
    """Build the package."""
    subprocess.check_call([sys.executable, "-m", "build"])
    print("  Package built")


def check_package():
    """Check package metadata."""
    result = subprocess.run(
        [sys.executable, "-m", "twine", "check", "dist/*"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR: Package check failed")
        print(result.stderr)
        sys.exit(1)


def upload_to_pypi():
    """Upload to PyPI."""
    subprocess.check_call([sys.executable, "-m", "twine", "upload", "dist/*"])
    print("  Uploaded to PyPI")


def main():
    """Main publish workflow."""
    print("=== FanFu PyPI Upload ===")

    print("\n[1/6] Bumping patch version...")
    new_version = bump_version()

    print("\n[2/6] Cleaning old builds...")
    clean_builds()

    print("\n[3/6] Installing build tools...")
    install_tools()

    print("\n[4/6] Building package...")
    build_package()

    print("\n[5/6] Checking package...")
    check_package()

    print("\n[6/6] Uploading to PyPI...")
    upload_to_pypi()

    print(f"\n=== Done! FanFu v{new_version} published ===")


if __name__ == "__main__":
    main()
