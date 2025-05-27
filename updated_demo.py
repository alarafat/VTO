"""
Main demo file with proper IDM-VTON dependency management
"""
import os
import sys
from pathlib import Path

# Add the current directory to path so we can import our dependency manager
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Import our dependency manager
try:
    from dependency_manager import setup_idmvton
    print("Dependency manager imported successfully")
except ImportError as e:
    print(f"Failed to import dependency manager: {e}")
    print("Make sure dependency_manager.py is in the same directory as demo.py")
    sys.exit(1)

def main():
    print("=== VTO Demo Starting ===")
    
    # Setup IDM-VTON dependency
    print("Setting up IDM-VTON dependency...")
    idmvton_manager = setup_idmvton(
        project_root=current_dir,
        install_deps=False  # Set to True if you want to auto-install packages
    )
    
    if not idmvton_manager:
        print("Failed to setup IDM-VTON dependency")
        print("Please check your dependencies/IDM-VTON folder structure")
        return
    
    print("IDM-VTON setup complete!")
    
    # Now try to import IDM-VTON modules
    print("\nImporting IDM-VTON modules...")
    
    # Try different ways to import apply_net
    apply_net = None
    import_attempts = [
        "gradio_demo.apply_net",
        "apply_net", 
        "gradio_demo"
    ]
    
    for module_name in import_attempts:
        try:
            if module_name == "gradio_demo":
                import gradio_demo
                if hasattr(gradio_demo, 'apply_net'):
                    apply_net = gradio_demo.apply_net
                    print(f"✓ Successfully imported via {module_name}")
                    break
            else:
                apply_net = idmvton_manager.get_module(module_name)
                if apply_net:
                    print(f"✓ Successfully imported {module_name}")
                    break
        except Exception as e:
            print(f"✗ Failed to import {module_name}: {e}")
            continue
    
    if not apply_net:
        print("Failed to import apply_net module")
        print("Available modules in IDM-VTON:")
        # List available modules
        idmvton_dir = current_dir / "dependencies" / "IDM-VTON"
        for item in idmvton_dir.rglob("*.py"):
            print(f"  {item.relative_to(idmvton_dir)}")
        return
    
    # Try to import other modules you might need
    other_modules = {}
    module_imports = [
        ("tryon_pipeline", "src.tryon_pipeline"),
        ("unet_hacked_garmnet", "src.unet_hacked_garmnet"), 
        ("unet_hacked_tryon", "src.unet_hacked_tryon")
    ]
    
    for name, module_path in module_imports:
        try:
            other_modules[name] = idmvton_manager.get_module(module_path)
            if other_modules[name]:
                print(f"✓ Successfully imported {name}")
        except Exception as e:
            print(f"✗ Failed to import {name}: {e}")
    
    print(f"\nSuccessfully imported {len([m for m in other_modules.values() if m is not None])} additional modules")
    
    # Your actual demo code would go here
    print("\n=== Running your demo logic ===")
    # For example:
    # result = apply_net.some_function()
    # print(f"Result: {result}")
    
    print("Demo completed successfully!")

if __name__ == "__main__":
    main()
