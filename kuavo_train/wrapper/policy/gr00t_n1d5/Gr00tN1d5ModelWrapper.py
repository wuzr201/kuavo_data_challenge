import os
from typing import Dict, Any
from torch import Tensor
import torch.nn as nn

from kuavo_train.wrapper.policy.gr00t_n1d5.Gr00tN1d5ConfigWrapper import CustomGr00tN1d5ConfigWrapper
from lerobot.policies.groot.groot_n1 import GR00TN15
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from kuavo_train.wrapper.policy.gr00t_n1d5.action_head.flow_matching_action_head import (
    FlowmatchingActionHead as KuavoFlowmatchingActionHead,
    FlowmatchingActionHeadConfig as KuavoFlowmatchingActionHeadConfig,
)


class CustomGr00tN1d5ModelWrapper(nn.Module):
    """
    Model wrapper for GR00T N1.5 that wraps GR00TN15.
    
    This wrapper loads the pretrained GR00T model and provides a consistent
    interface within the training framework. The model is loaded via
    GR00TN15.from_pretrained() which handles downloading and initialization.
    
    We use composition instead of inheritance because GR00TN15.from_pretrained()
    returns an already-initialized model instance.
    """
    
    def __init__(self, config: CustomGr00tN1d5ConfigWrapper, rtc_processor: RTCProcessor | None = None):
        """
        Initialize the GR00T model wrapper by loading from pretrained.
        
        Args:
            config: Custom GR00T configuration wrapper
            rtc_processor: Optional RTC processor for real-time chunking
        """
        super().__init__()
        
        # Handle Flash Attention compatibility issues
        self._handle_flash_attention_compatibility()
        
        # Load pretrained model using GR00TN15.from_pretrained
        # This will download and initialize the model with the specified tuning options
        base_model_path = getattr(config, 'base_model_path', 'nvidia/GR00T-N1.5-3B')
        
        self.model = GR00TN15.from_pretrained(
            pretrained_model_name_or_path=base_model_path,
            tune_llm=getattr(config, 'tune_llm', False),
            tune_visual=getattr(config, 'tune_visual', False),
            tune_projector=getattr(config, 'tune_projector', True),
            tune_diffusion_model=getattr(config, 'tune_diffusion_model', True),
            ignore_mismatched_sizes=True,
        )

        # Replace default action head with local enhanced multi-head version,
        # reusing the base config dict and letting Kuavo config add extra fields with defaults.
        try:
            ah_cfg_dict = dict(self.model.config.action_head_cfg)
            kuavo_cfg = KuavoFlowmatchingActionHeadConfig(**ah_cfg_dict)
            self.model.action_head = KuavoFlowmatchingActionHead(kuavo_cfg, rtc_processor=rtc_processor)
        except Exception as e:
            print(f"[GROOT] Failed to swap to enhanced Kuavo action head, falling back to base head: {e}")
        
        # Set compute dtype
        if hasattr(config, 'use_bf16') and config.use_bf16:
            self.model.compute_dtype = "bfloat16"
            self.model.config.compute_dtype = "bfloat16"
        
        # Store config reference
        self.wrapper_config = config
    
    def _handle_flash_attention_compatibility(self) -> None:
        """Handle Flash Attention compatibility issues by setting environment variables.
        
        This addresses the common 'undefined symbol' error that occurs when Flash Attention
        is compiled against a different PyTorch version than what's currently installed.
        """
        # Set environment variables to handle Flash Attention compatibility
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")
        
        # Try to import flash_attn and handle failures gracefully
        try:
            import flash_attn
            print(f"[GROOT] Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            print(f"[GROOT] Flash Attention not available: {e}")
            print("[GROOT] Will use fallback attention mechanism")
        except Exception as e:
            if "undefined symbol" in str(e):
                print(f"[GROOT] Flash Attention compatibility issue detected: {e}")
                print("[GROOT] This is likely due to PyTorch/Flash Attention version mismatch")
                print("[GROOT] Consider reinstalling Flash Attention with compatible version:")
                print("  pip uninstall flash-attn")
                print("  pip install --no-build-isolation flash-attn==2.6.3")
                print("[GROOT] Continuing with fallback attention mechanism")
            else:
                print(f"[GROOT] Flash Attention error: {e}")
                print("[GROOT] Continuing with fallback attention mechanism")
    
    def forward(self, inputs: Dict[str, Tensor]) -> Any:
        """
        Forward pass for training.
        
        Args:
            inputs: Input dictionary containing state, action, and eagle_* keys
            
        Returns:
            BatchFeature containing loss and other metrics
        """
        return self.model.forward(inputs)
    
    def get_action(self, inputs: Dict[str, Tensor], **kwargs) -> Any:
        """
        Get action predictions for inference.
        
        Args:
            inputs: Input dictionary containing state and eagle_* keys
            **kwargs: Additional arguments for action selection
            
        Returns:
            BatchFeature containing action_pred
        """
        return self.model.get_action(inputs, **kwargs)
    
    def _rtc_enabled(self) -> bool:
        """Check if RTC is enabled. GR00TN15 may not have this attr (e.g. older lerobot)."""
        if hasattr(self.model, "_rtc_enabled") and callable(getattr(self.model, "_rtc_enabled")):
            return self.model._rtc_enabled()
        return False
    
    # Delegate other important methods/attributes to the underlying model
    @property
    def backbone(self):
        return self.model.backbone
    
    @property
    def action_head(self):
        return self.model.action_head
    
    @property
    def action_horizon(self):
        return self.model.action_horizon
    
    @property
    def action_dim(self):
        return self.model.action_dim
    
    @property
    def compute_dtype(self):
        return self.model.compute_dtype
    
    @compute_dtype.setter
    def compute_dtype(self, value):
        self.model.compute_dtype = value
        if hasattr(self.model, 'config'):
            self.model.config.compute_dtype = value
