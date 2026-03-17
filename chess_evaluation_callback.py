"""
Chess LLM Evaluation Callback for Training

This module provides a callback for running full evaluation during training
with VLLM server management.
"""

import os
import subprocess
import time
import gc
from pathlib import Path

import torch
import requests
from transformers import TrainerCallback

from evaluation_helpers.eval_config import EvalConfig
from run_evaluation import run_full_evaluation


class ChessLLMEvaluationCallback(TrainerCallback):
    """Callback to run full evaluation suite during training with VLLM server management"""
    
    def __init__(self, 
                 model, 
                 tokenizer,
                 checkpoint_dir,
                 eval_every_n_steps=100, 
                 output_dir="./eval_results_during_training",
                 vllm_port=8000,
                 batch_size=4,
                 grad_accum_steps=2,
                 warmup_steps=500):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir
        self.eval_every_n_steps = eval_every_n_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.vllm_port = vllm_port
        self.vllm_process = None
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.warmup_steps = warmup_steps
    
    def _cleanup_gpu(self):
        """Clean up GPU memory before starting VLLM"""
        
        # Move model to CPU
        self.model.cpu()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        time.sleep(2)
        
        # if torch.cuda.is_available():
            # allocated = torch.cuda.memory_allocated(0) / 1024**3
            # reserved = torch.cuda.memory_reserved(0) / 1024**3
            # print(f"GPU Memory After Cleanup- Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    def _restore_model_to_gpu(self):
        """Move model back to GPU for training"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # if torch.cuda.is_available():
            # allocated = torch.cuda.memory_allocated(0) / 1024**3
            # print(f"   GPU Memory - Allocated: {allocated:.2f} GB")
    
    def _start_vllm(self, model_path):
        """Start VLLM server as subprocess"""

        
        cmd = [
            "vllm", "serve",
            # "/home/dipam/miniconda3/envs/chesscomments/bin/vllm", "serve", # example if using different conda env for vllm
            model_path,
            "--tokenizer", model_path,
            "--port", str(self.vllm_port),
            "--dtype", "bfloat16",
            "--max-model-len", "1200",
            "--gpu-memory-utilization", "0.5",
            "--enforce-eager",
            "--host", "0.0.0.0",
            "--disable-log-stats",
        ]

        try:
            log_file = open(f"./vllm_eval_{time.time()}.log", "w") 
            self.vllm_process = subprocess.Popen(
                cmd,
                # stdout=subprocess.DEVNULL,  # Or keep PIPE if you want logs
                stdout=log_file,  # Or keep PIPE if you want logs
                stderr=subprocess.STDOUT,
                text=True,
            )
            
            # Wait for server to be ready
            max_wait = 120  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                try:
                    response = requests.get(f"http://localhost:{self.vllm_port}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"VLLM server ready on port {self.vllm_port}")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(2)
            
            print("VLLM server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"Error starting VLLM: {e}")
            return False
    
    def _shutdown_vllm(self):
        """Shutdown VLLM server gracefully, with fallback to force kill"""
        # print("Shutting down VLLM server...")
        
        # Try graceful shutdown via API
        try:
            response = requests.post(
                f"http://localhost:{self.vllm_port}/shutdown",
                timeout=10
            )
            if response.status_code == 200:
                print("   Graceful shutdown successful")
                time.sleep(3)
                return True
        except Exception as e:
            print(f"   Graceful shutdown failed: {e}")
        
        # Fallback: terminate the process
        if self.vllm_process:
            try:
                # print("   Attempting process termination...")
                self.vllm_process.terminate()
                self.vllm_process.wait(timeout=10)
                # print("   Process terminated")
                return True
            except subprocess.TimeoutExpired:
                print("   Process termination timed out, force killing...")
                self.vllm_process.kill()
                self.vllm_process.wait()
                print("   Process killed")
        
        # Final fallback: pkill
        try:
            print("   Using pkill as final fallback...")
            subprocess.run(
                ["pkill", "-9", "-f", f"vllm.*{self.vllm_port}"],
                timeout=5
            )
            print("   pkill executed")
        except Exception as e:
            print(f"   pkill failed: {e}")
        
        time.sleep(3)
        return True
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        # Only evaluate at specified intervals
        if state.global_step % self.eval_every_n_steps != 0 and not state.global_step == self.warmup_steps:
            return
        
        # print("\n" + "="*80)
        print(f"Running Evaluation at step {state.global_step}")
        # print("="*80)
        
        # Save checkpoint for VLLM to load
        # checkpoint_path = f"{self.checkpoint_dir}/checkpoint-{state.global_step}"
        checkpoint_path = f"{self.checkpoint_dir}/temp_checkpoint"
        print(f"Saving checkpoint to {checkpoint_path}")
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        checkpoint_path = os.path.abspath(checkpoint_path)
        
        # Set model to eval mode and save training state
        was_training = self.model.training
        self.model.eval()
        
        try:
            # Step 1: Clean up GPU
            self._cleanup_gpu()
            
            # Step 2: Start VLLM
            if not self._start_vllm(checkpoint_path):
                print("Failed to start VLLM, skipping evaluation")
                return
            
            config = EvalConfig()
            config.model_name = checkpoint_path  # Point to the checkpoint
            config.output_dir = str(self.output_dir)  # Set output directory for results
            
            all_results = run_full_evaluation(config)
            
            log_entry = {"num_positions_trained": state.global_step * self.grad_accum_steps * self.batch_size}
            log_entry.update(all_results)
            
            # Import wandb here to avoid circular dependency
            import wandb
            if wandb.run is not None:  # Check if wandb is active
                wandb.log(log_entry, step=state.global_step)
            print(log_entry)
            state.log_history.append(log_entry)
            
        except Exception as e:
            print(f"Error during full evaluation: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Step 4: Shutdown VLLM
            self._shutdown_vllm()
            
            # Step 5: Restore model to GPU
            self._restore_model_to_gpu()
            
            # Restore training mode
            if was_training:
                self.model.train()
