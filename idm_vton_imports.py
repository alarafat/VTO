"""
Clean import interface for IDM-VTON modules
This module provides a simple way to import all IDM-VTON components
"""
import sys
from pathlib import Path
from dependency_manager import setup_idmvton

class IDMVTONImports:
    def __init__(self):
        self.manager = None
        self._imports_loaded = False
        self._load_imports()
    
    def _load_imports(self):
        """Load all IDM-VTON imports"""
        if self._imports_loaded:
            return
        
        print("Setting up IDM-VTON imports...")
        
        # Setup dependency manager
        self.manager = setup_idmvton()
        if not self.manager:
            raise ImportError("Failed to setup IDM-VTON dependency manager")
        
        # Core pipeline imports
        self._load_core_components()
        
        # Preprocessing imports  
        self._load_preprocessing_components()
        
        # External library imports
        self._load_external_components()
        
        self._imports_loaded = True
        print("IDM-VTON imports loaded successfully!")
    
    def _load_core_components(self):
        """Load core IDM-VTON components"""
        # TryonPipeline
        try:
            src_tryon = self.manager.get_module("src.tryon_pipeline")
            self.TryonPipeline = getattr(src_tryon, 'StableDiffusionXLInpaintPipeline', None)
        except Exception as e:
            print(f"Warning: Could not load TryonPipeline: {e}")
            self.TryonPipeline = None
        
        # UNet models
        try:
            src_garmnet = self.manager.get_module("src.unet_hacked_garmnet")
            self.UNet2DConditionModel_ref = getattr(src_garmnet, 'UNet2DConditionModel', None)
        except Exception as e:
            print(f"Warning: Could not load UNet2DConditionModel_ref: {e}")
            self.UNet2DConditionModel_ref = None
        
        try:
            src_tryon_unet = self.manager.get_module("src.unet_hacked_tryon")
            self.UNet2DConditionModel = getattr(src_tryon_unet, 'UNet2DConditionModel', None)
        except Exception as e:
            print(f"Warning: Could not load UNet2DConditionModel: {e}")
            self.UNet2DConditionModel = None
        
        # Gradio demo
        try:
            self.apply_net = self.manager.get_module("gradio_demo.apply_net")
        except Exception as e:
            print(f"Warning: Could not load apply_net: {e}")
            self.apply_net = None
    
    def _load_preprocessing_components(self):
        """Load preprocessing components"""
        # Human parsing
        try:
            parsing_module = self.manager.get_module("preprocess.humanparsing.run_parsing")
            self.Parsing = getattr(parsing_module, 'Parsing', None)
        except Exception as e:
            print(f"Warning: Could not load Parsing: {e}")
            self.Parsing = None
        
        # OpenPose
        try:
            openpose_module = self.manager.get_module("preprocess.openpose.run_openpose") 
            self.OpenPose = getattr(openpose_module, 'OpenPose', None)
        except Exception as e:
            print(f"Warning: Could not load OpenPose: {e}")
            self.OpenPose = None
    
    def _load_external_components(self):
        """Load external library components"""
        # Diffusers
        try:
            from diffusers import DDPMScheduler, AutoencoderKL
            self.DDPMScheduler = DDPMScheduler
            self.AutoencoderKL = AutoencoderKL
        except ImportError as e:
            print(f"Warning: Could not load diffusers components: {e}")
            self.DDPMScheduler = None
            self.AutoencoderKL = None
        
        # Detectron2
        try:
            from detectron2.data.detection_utils import convert_PIL_to_numpy, apply_exif_orientation
            self.convert_PIL_to_numpy = convert_PIL_to_numpy
            self.apply_exif_orientation = apply_exif_orientation
        except ImportError as e:
            print(f"Warning: Could not load detectron2 utilities: {e}")
            self.convert_PIL_to_numpy = None
            self.apply_exif_orientation = None
    
    def get_available_imports(self):
        """Get a dictionary of all available imports"""
        imports = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if attr_value is not None and attr_name != 'manager':
                    imports[attr_name] = attr_value
        return imports
    
    def print_import_status(self):
        """Print the status of all imports"""
        print("\n=== IDM-VTON Import Status ===")
        
        components = {
            'Core Components': [
                'TryonPipeline', 'UNet2DConditionModel_ref', 'UNet2DConditionModel', 'apply_net'
            ],
            'Preprocessing': [
                'Parsing', 'OpenPose'
            ],
            'External Libraries': [
                'DDPMScheduler', 'AutoencoderKL', 'convert_PIL_to_numpy', 'apply_exif_orientation'
            ]
        }
        
        for category, component_list in components.items():
            print(f"\n{category}:")
            for component in component_list:
                status = "✓" if getattr(self, component, None) is not None else "✗"
                print(f"  {status} {component}")

# Global instance for easy importing
_idmvton_imports = None

def get_idmvton_imports():
    """Get the global IDM-VTON imports instance"""
    global _idmvton_imports
    if _idmvton_imports is None:
        _idmvton_imports = IDMVTONImports()
    return _idmvton_imports

# Convenience functions for direct imports
def import_all():
    """Import all IDM-VTON components and return them as a namespace object"""
    imports = get_idmvton_imports()
    return imports

def import_core():
    """Import only core components"""
    imports = get_idmvton_imports()
    
    class CoreImports:
        def __init__(self):
            self.TryonPipeline = imports.TryonPipeline
            self.UNet2DConditionModel_ref = imports.UNet2DConditionModel_ref  
            self.UNet2DConditionModel = imports.UNet2DConditionModel
            self.apply_net = imports.apply_net
            self.DDPMScheduler = imports.DDPMScheduler
            self.AutoencoderKL = imports.AutoencoderKL
    
    return CoreImports()

def import_preprocessing():
    """Import only preprocessing components"""
    imports = get_idmvton_imports()
    
    class PreprocessingImports:
        def __init__(self):
            self.Parsing = imports.Parsing
            self.OpenPose = imports.OpenPose
            self.convert_PIL_to_numpy = imports.convert_PIL_to_numpy
            self.apply_exif_orientation = imports.apply_exif_orientation
    
    return PreprocessingImports()

# For backward compatibility - direct module-level imports
def setup_module_imports():
    """Setup module-level imports for backward compatibility"""
    imports = get_idmvton_imports()
    
    # Add to current module's globals
    current_module = sys.modules[__name__]
    for name, value in imports.get_available_imports().items():
        setattr(current_module, name, value)

# Auto-setup when imported
try:
    setup_module_imports()
except Exception as e:
    print(f"Warning: Could not setup module-level imports: {e}")
