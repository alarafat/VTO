#!/usr/bin/env python3
"""
Setup script for VTO project dependencies
Run this once to setup your environment
"""
import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_package(package, extra_args=None):
    """Install a Python package"""
    cmd = [sys.executable, "-m", "pip", "install", package]
    if extra_args:
        cmd.extend(extra_args)
    
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def setup_base_packages():
    """Install base packages required for IDM-VTON"""
    print("Installing base packages...")
    
    packages = [
        "torch",
        "torchvision", 
        "torchaudio",
        "transformers",
        "diffusers",
        "accelerate",
        "opencv-python",
        "pillow",
        "numpy",
        "scipy",
        "scikit-image",
        "gradio",
        "huggingface_hub"
    ]
    
    success_count = 0
    for package in packages:
        print(f"  Installing {package}...", end=" ")
        if install_package(package):
            print("✓")
            success_count += 1
        else:
            print("✗")
    
    print(f"Installed {success_count}/{len(packages)} base packages")
    return success_count == len(packages)

def setup_detectron2():
    """Install detectron2 with fallback options"""
    print("Installing detectron2...")
    
    # Try different installation methods
    install_commands = [
        # CUDA 11.8
        ["pip", "install", "detectron2", "-f", "https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html"],
        # CPU only
        ["pip", "install", "detectron2", "-f", "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html"],
        # From source (last resort)
        ["pip", "install", "git+https://github.com/facebookresearch/detectron2.git"]
    ]
    
    for i, cmd in enumerate(install_commands):
        try:
            print(f"  Trying method {i+1}/3...", end=" ")
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✓")
            return True
        except subprocess.CalledProcessError:
            print("✗")
            continue
    
    print("Failed to install detectron2 - you may need to install it manually")
    return False

def verify_dependencies_structure():
    """Verify the dependencies folder structure"""
    print("Verifying dependencies structure...")
    
    current_dir = Path(__file__).parent
    deps_dir = current_dir / "dependencies"
    
    if not deps_dir.exists():
        print(f"Creating dependencies directory: {deps_dir}")
        deps_dir.mkdir(exist_ok=True)
    
    # Check IDM-VTON
    idmvton_dir = deps_dir / "IDM-VTON"
    if idmvton_dir.exists():
        print(f"✓ IDM-VTON found at: {idmvton_dir}")
        
        # Check for key subdirectories
        key_dirs = ["src", "gradio_demo", "preprocess"]
        for subdir in key_dirs:
            subdir_path = idmvton_dir / subdir
            if subdir_path.exists():
                print(f"  ✓ {subdir}/")
            else:
                print(f"  ✗ {subdir}/ (missing)")
    else:
        print(f"✗ IDM-VTON not found at: {idmvton_dir}")
        print("  Please clone IDM-VTON into the dependencies folder:")
        print("  git clone https://github.com/yisol/IDM-VTON.git dependencies/IDM-VTON")
    
    # Check SAM2
    sam2_dir = deps_dir / "sam2"
    if sam2_dir.exists():
        print(f"✓ SAM2 found at: {sam2_dir}")
    else:
        print(f"✗ SAM2 not found at: {sam2_dir}")
        print("  Please clone SAM2 into the dependencies folder if needed")
    
    return idmvton_dir.exists()

def create_init_files():
    """Create __init__.py files where needed"""
    print("Creating __init__.py files...")
    
    current_dir = Path(__file__).parent
    deps_dir = current_dir / "dependencies"
    
    # Directories that might need __init__.py
    dirs_needing_init = [
        deps_dir,
        deps_dir / "IDM-VTON",
        deps_dir / "IDM-VTON" / "src",
        deps_dir / "IDM-VTON" / "gradio_demo",
        deps_dir / "sam2"
    ]
    
    for dir_path in dirs_needing_init:
        if dir_path.exists():
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"  Created: {init_file.relative_to(current_dir)}")

def test_imports():
    """Test if we can import everything"""
    print("Testing imports...")
    
    # Test basic packages
    test_packages = [
        "torch",
        "transformers", 
        "diffusers",
        "cv2",
        "PIL",
        "numpy"
    ]
    
    for package in test_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
    
    # Test detectron2 separately
    try:
        import detectron2
        print(f"  ✓ detectron2 (version {detectron2.__version__})")
    except ImportError:
        print("  ✗ detectron2")

def main():
    print("=== VTO Project Setup ===")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Verify dependencies structure
    if not verify_dependencies_structure():
        print("\nPlease fix the dependencies structure before continuing")
        return False
    
    # Create __init__.py files
    create_init_files()
    
    # Install packages
    print("\n=== Installing Packages ===")
    base_success = setup_base_packages()
    detectron2_success = setup_detectron2()
    
    # Test imports
    print("\n=== Testing Imports ===")
    test_imports()
    
    print("\n=== Setup Complete ===")
    if base_success and detectron2_success:
        print("✓ All packages installed successfully!")
        print("You can now run: python demo.py")
    else:
        print("⚠ Some packages failed to install")
        print("You may need to install them manually")
    
    return True

if __name__ == "__main__":
    main()
