import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Any
import wandb
from pathlib import Path
import json
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False
    
    # Distributed training
    distributed: bool = True
    local_rank: int = 0
    world_size: int = 1
    
    # Checkpointing
    save_steps: int = 1000
    save_dir: str = "checkpoints"
    
    # Logging
    log_steps: int = 100
    wandb_project: str = "efficient_transformer"
    
    # Optimization
    gradient_checkpointing: bool = True
    fsdp: bool = True  # Fully Sharded Data Parallel
    activation_checkpointing: bool = True

class DistributedTrainer:
    """Efficient distributed training with modern optimizations"""
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        config: TrainingConfig,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        
        # Setup distributed training
        if config.distributed:
            if config.fsdp:
                self.model = self._setup_fsdp(model)
            else:
                self.model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[config.local_rank],
                    output_device=config.local_rank
                )
        
        # Optimization
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if config.fp16 else None
        
        # Logging
        if config.local_rank == 0:
            wandb.init(project=config.wandb_project)
            
        # Save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_fsdp(self, model: nn.Module) -> nn.Module:
        """Setup Fully Sharded Data Parallel training"""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel,
            MixedPrecision,
            BackwardPrefetch,
            CPUOffload,
        )
        from torch.distributed.fsdp.wrap import (
            transformer_auto_wrap_policy,
            size_based_auto_wrap_policy,
        )
        
        # Mixed precision policy
        mixed_precision_policy = None
        if self.config.fp16:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        elif self.config.bf16:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            
        # FSDP configuration
        return FullyShardedDataParallel(
            model,
            mixed_precision=mixed_precision_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            cpu_offload=CPUOffload(offload_params=True),
            device_id=torch.cuda.current_device(),
            auto_wrap_policy=transformer_auto_wrap_policy,
        )
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay fix"""
        from torch.optim import AdamW
        
        # Separate weight decay parameters
        decay_parameters = []
        no_decay_parameters = []
        
        for module in self.model.modules():
            for name, param in module.named_parameters(recurse=False):
                if any(x in name for x in ['bias', 'LayerNorm', 'layer_norm', 'norm']):
                    no_decay_parameters.append(param)
                else:
                    decay_parameters.append(param)
                    
        optimizer_grouped_parameters = [
            {'params': decay_parameters, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_parameters, 'weight_decay': 0.0}
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with warmup"""
        from transformers import get_cosine_schedule_with_warmup
        
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
        )
        
    def save_checkpoint(self, step: int, metrics: Dict[str, float]):
        """Save training checkpoint"""
        if self.config.local_rank == 0:
            checkpoint = {
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': metrics,
            }
            
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                
            path = self.save_dir / f'checkpoint-{step}.pt'
            torch.save(checkpoint, path)
            
            # Save config
            with open(self.save_dir / 'config.json', 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
                
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        return checkpoint['step'], checkpoint['metrics']
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with optimizations"""
        # Forward pass with mixed precision
        with autocast(enabled=self.config.fp16 or self.config.bf16):
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
        # Backward pass with gradient scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        return {'loss': loss.item() * self.config.gradient_accumulation_steps}
        
    def optimizer_step(self):
        """Optimizer step with gradient clipping"""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        self.scheduler.step()
        self.optimizer.zero_grad()
        
    def train(self):
        """Main training loop with optimizations"""
        self.model.train()
        
        # Training loop
        step = 0
        accumulated_loss = 0
        
        with tqdm(total=self.config.max_steps, disable=self.config.local_rank != 0) as pbar:
            while step < self.config.max_steps:
                for batch in self.train_dataloader:
                    # Move batch to device
                    batch = {k: v.cuda() for k, v in batch.items()}
                    
                    # Training step
                    metrics = self.train_step(batch)
                    accumulated_loss += metrics['loss']
                    
                    # Gradient accumulation
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        self.optimizer_step()
                        
                        # Logging
                        if step % self.config.log_steps == 0 and self.config.local_rank == 0:
                            avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                            wandb.log({
                                'loss': avg_loss,
                                'learning_rate': self.scheduler.get_last_lr()[0],
                                'step': step,
                            })
                            accumulated_loss = 0
                            
                        # Checkpointing
                        if step % self.config.save_steps == 0:
                            self.save_checkpoint(step, {'loss': avg_loss})
                            
                        # Update progress
                        if self.config.local_rank == 0:
                            pbar.update(1)
                            pbar.set_postfix({'loss': avg_loss})
                            
                    step += 1
                    if step >= self.config.max_steps:
                        break
                        
        # Final checkpoint
        if self.config.local_rank == 0:
            self.save_checkpoint(step, {'loss': accumulated_loss / self.config.gradient_accumulation_steps})
            
    def evaluate(self) -> Dict[str, float]:
        """Evaluation loop"""
        if self.eval_dataloader is None:
            return {}
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, disable=self.config.local_rank != 0):
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs['loss'].item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        
        if self.config.local_rank == 0:
            wandb.log({'eval_loss': avg_loss})
            
        return {'loss': avg_loss}

def create_trainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    **kwargs: Any,
) -> DistributedTrainer:
    """Create trainer with configuration"""
    config = TrainingConfig(**kwargs)
    return DistributedTrainer(model, train_dataloader, eval_dataloader, config) 