"""
Optimized Depth-Adaptive and Learning Rate Scheduling

Key Optimizations:
- Vectorized schedule computation (no Python loops)
- Pre-computed and cached schedules
- GPU-native tensor operations
- Efficient parameter updates
- Dictionary-based dispatch (O(1) lookup)
- Type-safe implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union

class DepthScheduler(nn.Module):
    """
    OPTIMIZED: Layer-wise equivariance weight scheduling

    Supports multiple scheduling strategies:
    - constant: Same weight for all layers
    - exponential: Exponentially decaying weights
    - linear: Linearly decaying weights
    - inverse: Inverse decay pattern
    - u_shaped: U-shaped weight distribution
    - learnable: Learned weights with softplus activation
    - linear_inc: Linearly increasing weights (focus on deep layers)
    - exp_inc: Exponentially increasing weights (focus on deep layers)
    """

    def __init__(
        self,
        num_layers: int,
        schedule_type: str = 'exponential',
        alpha_0: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.1
    ):
        """
        Args:
            num_layers: Number of layers in the model
            schedule_type: Type of scheduling strategy
            alpha_0: Initial/maximum weight (or scaling factor)
            beta: Decay rate for exponential/inverse schedules
            gamma: Decay rate for linear schedule
        """
        super().__init__()
        self.num_layers = num_layers
        self.schedule_type = schedule_type.lower()
        self.alpha_0 = alpha_0
        self.beta = beta
        self.gamma = gamma

        # Validate schedule type
        valid_types = {
            'constant', 'exponential', 'linear', 'inverse', 'u_shaped', 'learnable',
            'linear_inc', 'exp_inc'
        }
        
        if self.schedule_type not in valid_types:
            # Fallback to constant if unknown, but warn or raise
            # raising is safer for debugging
            raise ValueError(f"schedule_type must be one of {valid_types}, got {schedule_type}")

        # Initialize or register buffer
        if self.schedule_type == 'learnable':
            # Initialize learnable parameters
            init_values = alpha_0 * np.exp(-beta * np.arange(num_layers))
            self.alpha = nn.Parameter(torch.tensor(init_values, dtype=torch.float32))
        else:
            # Pre-compute fixed schedules (vectorized)
            self._compute_schedule_vectorized()

    def _compute_schedule_vectorized(self):
        """
        OPTIMIZED: Vectorized schedule computation (no Python loops)
        Pre-computes all schedules at initialization
        """
        layer_indices = torch.arange(self.num_layers, dtype=torch.float32)

        if self.schedule_type == 'constant':
            # Constant: α_l = α_0
            alpha = torch.full((self.num_layers,), self.alpha_0, dtype=torch.float32)

        elif self.schedule_type == 'exponential':
            # Exponential: α_l = α_0 * exp(-β*l)
            alpha = self.alpha_0 * torch.exp(-self.beta * layer_indices)
            
        elif self.schedule_type == 'exp_inc':
            # Exponential Increasing: α_l = α_0 * exp(β * (l - (N-1))) 
            # Normalized so the last layer is alpha_0
            # Or simply reverse decay:
            reverse_indices = (self.num_layers - 1) - layer_indices
            alpha = self.alpha_0 * torch.exp(-self.beta * reverse_indices)

        elif self.schedule_type == 'linear':
            # Linear: α_l = max(0, α_0 - γ*l)
            alpha = torch.clamp(self.alpha_0 - self.gamma * layer_indices, min=0.0)
            
        elif self.schedule_type == 'linear_inc':
            # Linear Increasing: α_l = (l / (N-1)) * alpha_0
            if self.num_layers > 1:
                alpha = (layer_indices / (self.num_layers - 1)) * self.alpha_0
            else:
                alpha = torch.tensor([self.alpha_0])

        elif self.schedule_type == 'inverse':
            # Inverse: α_l = α_0 / (1 + β*l)
            alpha = self.alpha_0 / (1.0 + self.beta * layer_indices)

        elif self.schedule_type == 'u_shaped':
            # U-shaped: Distance from center
            mid = self.num_layers / 2.0
            distance_from_mid = torch.abs(layer_indices - (mid - 0.5)) / max(mid, 1.0)
            alpha = self.alpha_0 * (0.5 + 0.5 * distance_from_mid)

        # Register as buffer (not trainable, moved to device automatically)
        self.register_buffer('alpha', alpha, persistent=True)

    def get_alpha(self, layer_idx: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        OPTIMIZED: Get weight for specific layer

        Efficient indexing without Python conditionals in hot path
        """
        if isinstance(layer_idx, int):
            # Ensure it's a tensor on correct device if needed for logic, 
            # but for simple indexing python int works on cuda tensor too.
            pass 

        if self.schedule_type == 'learnable':
            # Apply softplus activation to learnable parameters
            if isinstance(layer_idx, int):
                 return F.softplus(self.alpha[layer_idx])
            return F.softplus(self.alpha[layer_idx])
        else:
            # Direct indexing for fixed schedules (very fast)
            return self.alpha[layer_idx]

    def get_all_alphas(self) -> torch.Tensor:
        """
        OPTIMIZED: Get all layer weights at once (vectorized)

        Returns: [num_layers] tensor of weights
        """
        if self.schedule_type == 'learnable':
            return F.softplus(self.alpha)
        else:
            return self.alpha.clone()

    def forward(self, layer_idx: int) -> torch.Tensor:
        """Forward pass - get weight for layer"""
        return self.get_alpha(layer_idx)

    def extra_repr(self) -> str:
        """String representation"""
        return (f"num_layers={self.num_layers}, schedule_type='{self.schedule_type}', "
                f"alpha_0={self.alpha_0}, beta={self.beta}, gamma={self.gamma}")

# Simple helper for standalone usage (replaces the previous factory function)
def get_layer_weights(num_layers: int, strategy: str, decay_rate: float = 0.5, device='cpu') -> torch.Tensor:
    """
    Factory wrapper to maintain compatibility with previous instructions.
    Uses DepthScheduler internally.
    """
    # Map parameters roughly
    # decay_rate serves as beta proxy for simple calls
    # We assume decay_rate is like 0.5 per step, which means exp(-beta) = 0.5 -> beta = -ln(0.5) = 0.69
    import math
    beta_proxy = -math.log(max(decay_rate, 1e-6)) 
    
    scheduler = DepthScheduler(
        num_layers=num_layers,
        schedule_type=strategy,
        alpha_0=1.0,
        beta=beta_proxy,
        gamma=0.1 # default
    )
    
    weights = scheduler.get_all_alphas().to(device)
    return weights


class WarmupScheduler:
    """
    OPTIMIZED: Learning rate warmup scheduler

    Linearly warms up learning rate from warmup_start_lr to base_lr
    over warmup_epochs epochs.

    Key optimizations:
    - Vectorized learning rate computation
    - Efficient parameter group updates
    - Linear interpolation instead of Python loops
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        base_lr: float,
        warmup_start_lr: float = 1e-6
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of epochs to warm up
            base_lr: Target learning rate after warmup
            warmup_start_lr: Starting learning rate for warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_start_lr = warmup_start_lr
        self.current_epoch = 0

        # Validate
        if warmup_epochs <= 0:
            raise ValueError(f"warmup_epochs must be > 0, got {warmup_epochs}")
        if base_lr <= 0:
            raise ValueError(f"base_lr must be > 0, got {base_lr}")

    def step(self):
        """
        OPTIMIZED: Update learning rate for current epoch

        Linear warmup: lr(t) = warmup_start_lr + (base_lr - warmup_start_lr) * t / warmup_epochs

        Vectorized computation instead of per-iteration updates
        """
        if self.current_epoch < self.warmup_epochs:
            # Vectorized linear interpolation (no loop over param groups initially)
            progress = self.current_epoch / self.warmup_epochs
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress

            # Update all param groups efficiently
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.current_epoch += 1

    def is_warming_up(self) -> bool:
        """Check if still in warmup phase"""
        return self.current_epoch < self.warmup_epochs

    def get_lr(self) -> float:
        """Get current learning rate"""
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            return self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress
        return self.base_lr


class CompositeScheduler:
    """
    OPTIMIZED: Composite scheduler combining warmup + main scheduler

    Enables warmup followed by any learning rate schedule (step, cosine, etc.)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_scheduler: Optional[WarmupScheduler],
        main_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_scheduler: Warmup scheduler (can be None)
            main_scheduler: Main LR scheduler (can be None)
        """
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler

    def step(self):
        """Step through warmup then main scheduler"""
        if self.warmup_scheduler is not None and self.warmup_scheduler.is_warming_up():
            self.warmup_scheduler.step()
        elif self.main_scheduler is not None:
            self.main_scheduler.step()

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config,
    steps_per_epoch: Optional[int] = None
) -> Optional[Union[torch.optim.lr_scheduler.LRScheduler, WarmupScheduler, CompositeScheduler]]:
    """
    OPTIMIZED: Factory function for learning rate schedulers

    Supports multiple scheduling strategies with efficient dispatch

    Args:
        optimizer: PyTorch optimizer
        config: Configuration object with scheduler settings
        steps_per_epoch: Optional steps per epoch for step-level scheduling

    Returns:
        Scheduler object or None
    """

    # Dictionary-based dispatch (O(1) instead of O(N) if-elif chains)
    scheduler_factory = {
        'step': lambda: torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler.lr_step_size,
            gamma=config.scheduler.lr_gamma
        ),
        'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            eta_min=getattr(config.scheduler, 'eta_min', 1e-6)
        ),
        'cosine_warm_restarts': lambda: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.training.num_epochs // 4,
            T_mult=2,
            eta_min=getattr(config.scheduler, 'eta_min', 1e-6)
        ),
        'plateau': lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler.lr_gamma,
            patience=getattr(config.scheduler, 'plateau_patience', 10),
            verbose=getattr(config.scheduler, 'verbose', False)
        ),
        'exponential': lambda: torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.scheduler.lr_gamma
        ),
        'linear': lambda: torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=getattr(config.scheduler, 'start_factor', 1.0),
            total_iters=config.training.num_epochs
        ),
        'none': lambda: None
    }

    # Get main scheduler
    schedule_type = getattr(config.scheduler, 'lr_schedule', 'step').lower()

    if schedule_type not in scheduler_factory:
        raise ValueError(f"Unknown lr_schedule: {schedule_type}. "
                        f"Must be one of {list(scheduler_factory.keys())}")

    main_scheduler = scheduler_factory[schedule_type]()

    # Add warmup if enabled
    if getattr(config.scheduler, 'warmup_epochs', 0) > 0 and main_scheduler is not None:
        warmup_scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=config.scheduler.warmup_epochs,
            base_lr=config.training.learning_rate,
            warmup_start_lr=getattr(config.scheduler, 'warmup_start_lr', 1e-6)
        )
        return CompositeScheduler(optimizer, warmup_scheduler, main_scheduler)

    return main_scheduler


def get_depth_scheduler(
    num_layers: int,
    config
) -> DepthScheduler:
    """
    OPTIMIZED: Factory function for depth schedulers

    Args:
        num_layers: Number of layers in model
        config: Configuration object with scheduler settings

    Returns:
        DepthScheduler instance
    """
    schedule_type = getattr(config.scheduler, 'depth_schedule', 'exponential')
    alpha_0 = getattr(config.scheduler, 'alpha_0', 1.0)
    beta = getattr(config.scheduler, 'beta', 0.1)
    gamma = getattr(config.scheduler, 'gamma', 0.1)

    return DepthScheduler(
        num_layers=num_layers,
        schedule_type=schedule_type,
        alpha_0=alpha_0,
        beta=beta,
        gamma=gamma
    )