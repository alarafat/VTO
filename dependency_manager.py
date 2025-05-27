"""
Dependency Manager for IDM-VTON
This module handles the setup and import of IDM-VTON as a dependency
"""
import os
import sys
import importlib.util
from pathlib import Path

class IDMVTONDependency:
    def __init__(self, project_root=None):
        if project_root is None:
            # Get the directory where this script is located
            self.project_root = Path(__file__).parent.absolute()
        else:
            self.project_root = Path(project_root).absolute()
        
        self.dependencies_dir = self.project_root / "dependencies"
        self.idmvton_dir = self.dependencies_dir / "IDM-VTON"
        self.sam2_dir = self.dependencies_dir / "sam2"
        
        self._setup_complete = False
    
    def setup_paths(self):
        """Setup all necessary paths for IDM-VTON"""
        paths_to_add = []
        
        # Main IDM-VTON directory
        if self.idmvton_dir.exists():
            paths_to_add.append(str(self.idmvton_dir))
        
        # Common subdirectories in IDM-VTON
        potential_subdirs = [
            "src",
            "gradio_demo", 
            "preprocess",
            "utils"
        ]
        
        for subdir in potential_subdirs:
            subdir_path = self.idmvton_dir / subdir
            if subdir_path.exists():
                paths_to_add.append(str(subdir_path))
        
        # Add SAM2 if it exists
        if self.sam2_dir.exists():
            paths_to_add.append(str(self.sam2_dir))
        
        # Add paths to sys.path (insert at beginning for priority)
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
                print(f"Added to Python path: {path}")
        
        return paths_to_add
    
    def install_requirements(self):
        """Install required packages for IDM-VTON"""
        import subprocess
        
        required_packages = [
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
            "xformers"  # Optional but recommended
        ]
        
        # Special handling for detectron2
        detectron2_commands = [
            "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html",
            "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html"
        ]
        
        print("Installing basic requirements...")
        for package in required_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
        
        print("\nInstalling detectron2...")
        for cmd in detectron2_commands:
            try:
                subprocess.check_call(cmd.split())
                print("✓ Installed detectron2")
                break
            except subprocess.CalledProcessError:
                continue
        else:
            print("✗ Failed to install detectron2 - you may need to install it manually")
    
    def verify_structure(self):
        """Verify that IDM-VTON has the expected structure"""
        print(f"Checking IDM-VTON structure in: {self.idmvton_dir}")
        
        if not self.idmvton_dir.exists():
            print(f"✗ IDM-VTON directory not found: {self.idmvton_dir}")
            return False
        
        # Check for key files/directories
        expected_items = [
            "src",
            "gradio_demo", 
            "preprocess"
        ]
        
        found_items = []
        for item in expected_items:
            item_path = self.idmvton_dir / item
            if item_path.exists():
                found_items.append(item)
                print(f"✓ Found: {item}")
            else:
                print(f"✗ Missing: {item}")
        
        # List actual contents
        print(f"\nActual contents of {self.idmvton_dir}:")
        try:
            for item in self.idmvton_dir.iterdir():
                item_type = "📁" if item.is_dir() else "📄"
                print(f"  {item_type} {item.name}")
        except Exception as e:
            print(f"Error listing directory: {e}")
        
        return len(found_items) > 0
    
    def setup(self, install_deps=False):
        """Complete setup of IDM-VTON dependency"""
        print("=== Setting up IDM-VTON Dependency ===")
        
        # Verify structure
        if not self.verify_structure():
            print("Please ensure IDM-VTON is properly placed in the dependencies folder")
            return False
        
        # Setup paths
        added_paths = self.setup_paths()
        print(f"Added {len(added_paths)} paths to Python path")
        
        # Install dependencies if requested
        if install_deps:
            self.install_requirements()
        
        # Test imports
        self.test_imports()
        
        self._setup_complete = True
        return True
    
    def test_imports(self):
        """Test if we can import IDM-VTON modules"""
        print("\n=== Testing Imports ===")
        
        # Test basic imports
        test_modules = [
            "detectron2",
            "torch", 
            "transformers",
            "diffusers"
        ]
        
        for module in test_modules:
            try:
                importlib.import_module(module)
                print(f"✓ {module}")
            except ImportError as e:
                print(f"✗ {module}: {e}")
        
        # Test IDM-VTON specific imports
        idmvton_modules = []
        
        # Try to find common IDM-VTON modules
        potential_modules = [
            "gradio_demo.apply_net",
            "src.tryon_pipeline", 
            "preprocess.humanparsing.run_parsing",
            "preprocess.openpose.run_openpose"
        ]
        
        for module in potential_modules:
            try:
                importlib.import_module(module)
                print(f"✓ IDM-VTON: {module}")
                idmvton_modules.append(module)
            except ImportError:
                # Try without prefix
                try:
                    short_name = module.split('.')[-1]
                    importlib.import_module(short_name)
                    print(f"✓ IDM-VTON: {short_name}")
                    idmvton_modules.append(short_name)
                except ImportError:
                    print(f"✗ IDM-VTON: {module}")
        
        return len(idmvton_modules) > 0
    
    def get_module(self, module_name):
        """Safely import and return an IDM-VTON module"""
        if not self._setup_complete:
            self.setup()
        
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            print(f"Failed to import {module_name}: {e}")
            return None

# Convenience function for easy setup
def setup_idmvton(project_root=None, install_deps=False):
    """Easy setup function for IDM-VTON dependency"""
    manager = IDMVTONDependency(project_root)
    success = manager.setup(install_deps)
    return manager if success else None

# Auto-setup when imported
_global_manager = None

def get_idmvton_manager():
    """Get the global IDM-VTON dependency manager"""
    global _global_manager
    if _global_manager is None:
        _global_manager = setup_idmvton()
    return _global_manager
