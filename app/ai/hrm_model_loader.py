"""
HRM Model Loader

Properly loads trained HRM checkpoints and makes them available
for inference in the Elect_Engin1 application.
"""

import logging
import torch
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class HRMModelLoader:
    """Handles loading and managing HRM model checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "hrm_checkpoints"):
        """
        Initialize model loader
        
        Args:
            checkpoint_dir: Directory containing trained model checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.loaded_models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"HRM Model Loader initialized")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"  Device: {self.device}")
    
    def load_checkpoint(self, checkpoint_path: str, model_name: str) -> Optional[torch.nn.Module]:
        """
        Load a trained HRM checkpoint
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
            model_name: Name to cache this model under
            
        Returns:
            Loaded model in eval mode, or None if loading fails
        """
        ckpt_file = Path(checkpoint_path)
        
        if not ckpt_file.exists():
            logger.warning(f"Checkpoint not found: {ckpt_file}")
            return None
        
        try:
            logger.info(f"Loading {model_name} from {ckpt_file}...")
            
            # Load checkpoint
            checkpoint = torch.load(ckpt_file, map_location=self.device)
            
            # Extract model architecture from checkpoint
            if "config" not in checkpoint:
                logger.error(f"Checkpoint missing 'config' key")
                return None
            
            # Import model class
            from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
            
            # Create model from config
            model = HierarchicalReasoningModel_ACTV1(checkpoint["config"])
            
            # Load trained weights
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                logger.error("Checkpoint missing model weights")
                return None
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            # Cache loaded model
            self.loaded_models[model_name] = {
                "model": model,
                "config": checkpoint["config"],
                "checkpoint_path": str(ckpt_file),
                "metadata": checkpoint.get("metadata", {})
            }
            
            logger.info(f"✓ Successfully loaded {model_name}")
            logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            logger.info(f"  Device: {self.device}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}", exc_info=True)
            return None
    
    def get_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """
        Get a loaded model by name
        
        Args:
            model_name: Name of the model
            
        Returns:
            The model, or None if not loaded
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]["model"]
        return None
    
    def load_all_available(self) -> Dict[str, torch.nn.Module]:
        """
        Load all available checkpoints from checkpoint directory
        
        Returns:
            Dictionary of loaded models
        """
        if not self.checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory not found: {self.checkpoint_dir}")
            return {}
        
        logger.info(f"Scanning for checkpoints in {self.checkpoint_dir}...")
        
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        
        if not checkpoint_files:
            logger.warning(f"No .pt checkpoint files found in {self.checkpoint_dir}")
            return {}
        
        logger.info(f"Found {len(checkpoint_files)} checkpoint files")
        
        # Load each checkpoint
        for ckpt_file in checkpoint_files:
            model_name = ckpt_file.stem  # filename without .pt extension
            self.load_checkpoint(str(ckpt_file), model_name)
        
        logger.info(f"Loaded {len(self.loaded_models)} models successfully")
        
        return {name: info["model"] for name, info in self.loaded_models.items()}
    
    def predict(
        self,
        model_name: str,
        input_data: torch.Tensor,
        **kwargs
    ) -> Optional[torch.Tensor]:
        """
        Run inference with a loaded model
        
        Args:
            model_name: Name of the model to use
            input_data: Input tensor
            **kwargs: Additional arguments for model forward pass
            
        Returns:
            Model output, or None if model not available
        """
        model = self.get_model(model_name)
        
        if model is None:
            logger.error(f"Model '{model_name}' not loaded")
            return None
        
        try:
            with torch.no_grad():
                # Move input to device
                input_data = input_data.to(self.device)
                
                # Run inference
                output = model(input_data, **kwargs)
                
                return output
                
        except Exception as e:
            logger.error(f"Inference failed for {model_name}: {e}", exc_info=True)
            return None
    
    def get_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model"""
        if model_name in self.loaded_models:
            info = self.loaded_models[model_name].copy()
            # Remove model object for serialization
            info.pop("model", None)
            return info
        return None
    
    def list_loaded_models(self) -> list:
        """List all loaded model names"""
        return list(self.loaded_models.keys())


# Global loader instance
_model_loader = None

def get_model_loader(checkpoint_dir: str = "hrm_checkpoints") -> HRMModelLoader:
    """Get or create the global model loader"""
    global _model_loader
    if _model_loader is None:
        _model_loader = HRMModelLoader(checkpoint_dir)
    return _model_loader
