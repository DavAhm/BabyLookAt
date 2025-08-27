#!/usr/bin/env python3
"""
Enhanced SAM2 Video Analysis Tool - Optimized for Local PC Use
Major improvements:
- Better memory management for consumer GPUs
- Multi-threade d processing
- Real-time preview during processing
- Enhanced UI with progress tracking
- Batch processing support
- Improved error recovery
- Better file organization
"""

import os
import sys
import json
import yaml
import time
import shutil
import psutil
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue

# GUI imports
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import customtkinter as ctk  # Modern UI

# Computer vision imports
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk

# Data processing
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm import tqdm

# Video processing
import subprocess
import ffmpeg

# Logging
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()

print("🔧 Script is starting...")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sam2_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================== SYSTEM OPTIMIZATION ==================

class SystemOptimizer:
    """Optimize system resources for local PC use"""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_available = torch.cuda.is_available()
        self.gpu_memory_gb = 0
        
        if self.gpu_available:
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
    def get_optimal_settings(self):
        """Determine optimal settings based on system capabilities"""
        settings = {
            'device': 'cuda' if self.gpu_available else 'cpu',
            'batch_size': 1,
            'num_workers': 2,
            'prefetch_frames': 10,
            'cache_size': 100,
            'use_mixed_precision': False,
            'offload_to_cpu': True,
            'video_backend': 'opencv',  # or 'ffmpeg'
            'max_resolution': (1920, 1080)
        }
        
        # Adjust based on GPU memory
        if self.gpu_available:
            if self.gpu_memory_gb >= 8:
                settings['batch_size'] = 4
                settings['prefetch_frames'] = 30
                settings['cache_size'] = 200
                settings['use_mixed_precision'] = True
            elif self.gpu_memory_gb >= 4:
                settings['batch_size'] = 2
                settings['prefetch_frames'] = 20
                settings['cache_size'] = 150
            else:
                # Low GPU memory
                settings['batch_size'] = 1
                settings['prefetch_frames'] = 10
                settings['offload_to_cpu'] = True
                
        # Adjust based on CPU
        if self.cpu_count >= 8:
            settings['num_workers'] = 4
        elif self.cpu_count >= 4:
            settings['num_workers'] = 2
        else:
            settings['num_workers'] = 1
            
        # Adjust based on RAM
        if self.ram_gb < 8:
            settings['cache_size'] = 50
            settings['prefetch_frames'] = 5
            
        return settings
    
    def optimize_torch(self):
        """Apply PyTorch optimizations"""
        if self.gpu_available:
            # Enable TF32 on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn benchmarking for faster convolutions
            torch.backends.cudnn.benchmark = True
            
            # Set memory fraction to prevent OOM
            if self.gpu_memory_gb < 6:
                torch.cuda.set_per_process_memory_fraction(0.8)
            else:
                torch.cuda.set_per_process_memory_fraction(0.9)
                
        # CPU optimizations
        torch.set_num_threads(min(4, self.cpu_count))
        
        return True

# ================== ENHANCED VIDEO PROCESSOR ==================

class EnhancedVideoProcessor:
    """Optimized video processor with caching and threading"""
    
    def __init__(self, video_path, output_dir, settings):
        self.video_path = video_path
        self.output_dir = output_dir
        self.settings = settings
        self.frame_cache = deque(maxlen=settings['cache_size'])
        self.processing_queue = queue.Queue(maxsize=settings['prefetch_frames'])
        self.results_queue = queue.Queue()
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap.release()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=settings['num_workers'])
        
    def extract_frames_optimized(self, skip_frames=1, resize_factor=1.0):
        """Extract frames with optimization and optional downsampling"""
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Use ffmpeg for faster extraction
        if self.settings['video_backend'] == 'ffmpeg':
            return self._extract_frames_ffmpeg(frames_dir, skip_frames, resize_factor)
        else:
            return self._extract_frames_opencv(frames_dir, skip_frames, resize_factor)
            
    def _extract_frames_ffmpeg(self, frames_dir, skip_frames, resize_factor):
        """Extract frames using ffmpeg (faster)"""
        try:
            # Build ffmpeg command
            stream = ffmpeg.input(self.video_path)
            
            # Apply filters
            if skip_frames > 1:
                stream = stream.filter('select', f'not(mod(n,{skip_frames}))')
                
            if resize_factor != 1.0:
                new_width = int(self.width * resize_factor)
                new_height = int(self.height * resize_factor)
                stream = stream.filter('scale', new_width, new_height)
                
            # Output settings
            stream = stream.output(
                os.path.join(frames_dir, '%05d.jpg'),
                start_number=0,
                qscale=2
            )
            
            # Run ffmpeg
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            
            # Count extracted frames
            extracted_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            effective_fps = self.fps / skip_frames
            
            return effective_fps, extracted_frames
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            return -1, -1
            
    def _extract_frames_opencv(self, frames_dir, skip_frames, resize_factor):
        """Extract frames using OpenCV with progress bar"""
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        saved_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Extracting frames...", 
                total=self.frame_count // skip_frames
            )
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % skip_frames == 0:
                    # Resize if needed
                    if resize_factor != 1.0:
                        new_width = int(frame.shape[1] * resize_factor)
                        new_height = int(frame.shape[0] * resize_factor)
                        frame = cv2.resize(frame, (new_width, new_height), 
                                         interpolation=cv2.INTER_AREA)
                    
                    # Save frame
                    output_path = os.path.join(frames_dir, f"{saved_count:05d}.jpg")
                    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_count += 1
                    progress.update(task, advance=1)
                    
                frame_idx += 1
                
        cap.release()
        effective_fps = self.fps / skip_frames
        
        return effective_fps, saved_count

# ================== ENHANCED SAM2 PREDICTOR ==================

class EnhancedSAM2Predictor:
    """Enhanced SAM2 predictor with better memory management"""
    
    def __init__(self, model_cfg, checkpoint_path, device, settings):
        self.device = device
        self.settings = settings
        self.model_loaded = False
        
        # Lazy loading to save memory
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        
    def load_model(self):
        """Load model with optimizations"""
        if self.model_loaded:
            return
            
        try:
            from sam2.build_sam import build_sam2_video_predictor
            
            # Load with memory optimizations
            self.predictor = build_sam2_video_predictor(
                self.model_cfg, 
                self.checkpoint_path,
                device=self.device
            )
            
            # Apply mixed precision if enabled
            if self.settings['use_mixed_precision'] and self.device.type == 'cuda':
                self.predictor.model = self.predictor.model.half()
                
            self.model_loaded = True
            logger.info("SAM2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            raise
            
    def process_video_streaming(self, video_dir, prompts, callback=None):
        """Process video with streaming results"""
        if not self.model_loaded:
            self.load_model()
            
        # Initialize state
        inference_state = self.predictor.init_state(
            video_path=video_dir,
            offload_video_to_cpu=self.settings['offload_to_cpu'],
            offload_state_to_cpu=self.settings['offload_to_cpu'],
            async_loading_frames=True,
        )
        
        self.predictor.reset_state(inference_state)
        
        # Add prompts
        for obj_id, prompt_data in prompts.items():
            points = np.array(prompt_data['points'], dtype=np.float32)
            labels = np.array(prompt_data['labels'], dtype=np.int32)
            
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=prompt_data['frame_idx'],
                obj_id=obj_id,
                points=points,
                labels=labels,
            )
            
        # Process frames with streaming
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            frame_results = {}
            
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                if len(mask.shape) == 3:
                    mask = mask[0]
                frame_results[obj_id] = mask
                
            # Call callback with results
            if callback:
                callback(out_frame_idx, frame_results)
                
            yield out_frame_idx, frame_results
            
            # Periodic cleanup
            if out_frame_idx % 50 == 0:
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                
        # Cleanup
        self.predictor.reset_state(inference_state)
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None

# ================== MODERN UI APPLICATION ==================

class ModernVideoAnalysisApp:
    """Modern UI for SAM2 video analysis"""
    
    def __init__(self):
        # Initialize customtkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("SAM2 Video Analysis - Enhanced Edition")
        self.root.geometry("1200x800")
        
        # System optimizer
        self.optimizer = SystemOptimizer()
        self.settings = self.optimizer.get_optimal_settings()
        self.optimizer.optimize_torch()
        
        # Initialize components
        self.predictor = None
        self.current_project = None
        self.processing_thread = None
        
        # Setup UI
        self.setup_ui()
        self.load_predictor()
        
    def setup_ui(self):
        """Create modern UI layout"""
        # Main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        self.left_panel = ctk.CTkFrame(self.main_container, width=300)
        self.left_panel.pack(side="left", fill="y", padx=(0, 10))
        self.left_panel.pack_propagate(False)
        
        # Right panel - Preview and results
        self.right_panel = ctk.CTkFrame(self.main_container)
        self.right_panel.pack(side="right", fill="both", expand=True)
        
        # Setup panels
        self.setup_controls()
        self.setup_preview()
        self.setup_status_bar()
        
    def setup_controls(self):
        """Setup control panel"""
        # Title
        title = ctk.CTkLabel(
            self.left_panel, 
            text="SAM2 Video Analysis",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=(10, 20))
        
        # System info
        sys_frame = ctk.CTkFrame(self.left_panel)
        sys_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        sys_info = f"GPU: {'Available' if self.optimizer.gpu_available else 'Not Available'}\n"
        sys_info += f"RAM: {self.optimizer.ram_gb:.1f} GB\n"
        sys_info += f"CPU Cores: {self.optimizer.cpu_count}"
        
        sys_label = ctk.CTkLabel(sys_frame, text=sys_info, font=ctk.CTkFont(size=12))
        sys_label.pack(padx=10, pady=10)
        
        # Video selection
        self.video_frame = ctk.CTkFrame(self.left_panel)
        self.video_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(self.video_frame, text="Video Source:").pack(anchor="w", padx=5, pady=(5, 0))
        
        self.video_path_var = tk.StringVar()
        self.video_entry = ctk.CTkEntry(self.video_frame, textvariable=self.video_path_var)
        self.video_entry.pack(fill="x", padx=5, pady=5)
        
        btn_frame = ctk.CTkFrame(self.video_frame)
        btn_frame.pack(fill="x", padx=5, pady=(0, 5))
        
        self.browse_btn = ctk.CTkButton(
            btn_frame, 
            text="Browse", 
            command=self.browse_video,
            width=70
        )
        self.browse_btn.pack(side="left", padx=(0, 5))
        
        self.batch_btn = ctk.CTkButton(
            btn_frame,
            text="Batch Mode",
            command=self.batch_mode,
            width=70
        )
        self.batch_btn.pack(side="left")
        
        # Processing options
        self.options_frame = ctk.CTkFrame(self.left_panel)
        self.options_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(self.options_frame, text="Processing Options:").pack(anchor="w", padx=5, pady=(5, 0))
        
        # Skip frames
        skip_frame = ctk.CTkFrame(self.options_frame)
        skip_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(skip_frame, text="Process every").pack(side="left")
        self.skip_frames_var = tk.IntVar(value=1)
        self.skip_spin = ctk.CTkEntry(skip_frame, textvariable=self.skip_frames_var, width=50)
        self.skip_spin.pack(side="left", padx=5)
        ctk.CTkLabel(skip_frame, text="frames").pack(side="left")
        
        # Resize factor
        resize_frame = ctk.CTkFrame(self.options_frame)
        resize_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(resize_frame, text="Resize factor:").pack(side="left")
        self.resize_var = tk.DoubleVar(value=1.0)
        self.resize_slider = ctk.CTkSlider(
            resize_frame,
            from_=0.25,
            to=1.0,
            variable=self.resize_var,
            command=self.update_resize_label
        )
        self.resize_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.resize_label = ctk.CTkLabel(resize_frame, text="1.0x")
        self.resize_label.pack(side="left")
        
        # Detection options
        self.detection_frame = ctk.CTkFrame(self.left_panel)
        self.detection_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(self.detection_frame, text="Detection Settings:").pack(anchor="w", padx=5, pady=(5, 0))
        
        # Overlap threshold
        thresh_frame = ctk.CTkFrame(self.detection_frame)
        thresh_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(thresh_frame, text="Overlap threshold:").pack(side="left")
        self.threshold_var = tk.IntVar(value=10)
        self.threshold_slider = ctk.CTkSlider(
            thresh_frame,
            from_=1,
            to=50,
            variable=self.threshold_var,
            command=self.update_threshold_label
        )
        self.threshold_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.threshold_label = ctk.CTkLabel(thresh_frame, text="10%")
        self.threshold_label.pack(side="left")
        
        # Export options
        self.export_frame = ctk.CTkFrame(self.left_panel)
        self.export_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        ctk.CTkLabel(self.export_frame, text="Export Options:").pack(anchor="w", padx=5, pady=(5, 0))
        
        self.export_video_var = tk.BooleanVar(value=True)
        self.export_video_cb = ctk.CTkCheckBox(
            self.export_frame,
            text="Export annotated video",
            variable=self.export_video_var
        )
        self.export_video_cb.pack(anchor="w", padx=5, pady=2)
        
        self.export_csv_var = tk.BooleanVar(value=True)
        self.export_csv_cb = ctk.CTkCheckBox(
            self.export_frame,
            text="Export CSV data",
            variable=self.export_csv_var
        )
        self.export_csv_cb.pack(anchor="w", padx=5, pady=2)
        
        self.export_elan_var = tk.BooleanVar(value=True)
        self.export_elan_cb = ctk.CTkCheckBox(
            self.export_frame,
            text="Export ELAN file",
            variable=self.export_elan_var
        )
        self.export_elan_cb.pack(anchor="w", padx=5, pady=2)
        
        # Process button
        self.process_btn = ctk.CTkButton(
            self.left_panel,
            text="Process Video",
            command=self.process_video,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.process_btn.pack(fill="x", padx=10, pady=(0, 10))
        
        # Stop button
        self.stop_btn = ctk.CTkButton(
            self.left_panel,
            text="Stop Processing",
            command=self.stop_processing,
            height=30,
            fg_color="red",
            state="disabled"
        )
        self.stop_btn.pack(fill="x", padx=10)
        
    def setup_preview(self):
        """Setup preview panel"""
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Preview tab
        self.preview_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.preview_frame, text="Preview")
        
        self.preview_label = ctk.CTkLabel(
            self.preview_frame, 
            text="No video loaded",
            font=ctk.CTkFont(size=16)
        )
        self.preview_label.pack(expand=True)
        
        # Results tab
        self.results_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        
        # Progress tab
        self.progress_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.progress_frame, text="Progress")
        
        # Progress widgets
        self.progress_text = scrolledtext.ScrolledText(
            self.progress_frame,
            height=20,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.progress_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Metrics tab
        self.metrics_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.metrics_frame, text="Metrics")
        
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_frame = ctk.CTkFrame(self.root, height=30)
        self.status_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=10)
        
        # Memory usage
        self.memory_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.memory_label.pack(side="right", padx=10)
        
        # Start memory monitoring
        self.update_memory_usage()
        
    def update_memory_usage(self):
        """Update memory usage display"""
        try:
            # CPU memory
            cpu_percent = psutil.virtual_memory().percent
            
            # GPU memory
            if self.optimizer.gpu_available:
                gpu_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_percent = (gpu_used / gpu_total) * 100
                
                memory_text = f"CPU: {cpu_percent:.1f}% | GPU: {gpu_used:.1f}/{gpu_total:.1f} GB ({gpu_percent:.1f}%)"
            else:
                memory_text = f"CPU: {cpu_percent:.1f}%"
                
            self.memory_label.configure(text=memory_text)
            
        except:
            pass
            
        # Schedule next update
        self.root.after(2000, self.update_memory_usage)
        
    def update_resize_label(self, value):
        """Update resize label"""
        self.resize_label.configure(text=f"{float(value):.2f}x")
        
    def update_threshold_label(self, value):
        """Update threshold label"""
        self.threshold_label.configure(text=f"{int(value)}%")
        
    def browse_video(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.video_path_var.set(filename)
            self.load_video_preview(filename)
            
    def load_video_preview(self, video_path):
        """Load video preview"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            
            if ret:
                # Get middle frame
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret, frame = cap.read()
                
                if ret:
                    # Convert to RGB and resize for preview
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width = frame.shape[:2]
                    
                    # Calculate preview size
                    max_width = 600
                    max_height = 400
                    
                    scale = min(max_width / width, max_height / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert to PhotoImage
                    image = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(image)
                    
                    # Update preview
                    self.preview_label.configure(image=photo, text="")
                    self.preview_label.image = photo  # Keep reference
                    
                    # Update status
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps
                    
                    info_text = f"Video loaded: {width}x{height}, {fps:.1f} FPS, {duration:.1f}s ({total_frames} frames)"
                    self.status_label.configure(text=info_text)
                    
            cap.release()
            
        except Exception as e:
            logger.error(f"Failed to load video preview: {e}")
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")
            
    def batch_mode(self):
        """Open batch processing dialog"""
        batch_dialog = BatchProcessingDialog(self.root, self)
        
    def load_predictor(self):
        """Load SAM2 predictor"""
        try:
            # Check for model files
            checkpoint_path = "../checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            
            if not os.path.exists(checkpoint_path):
                checkpoint_path = "../checkpoints/sam2.1_hiera_small.pt"
                model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
                
            if not os.path.exists(checkpoint_path):
                messagebox.showerror(
                    "Model Not Found",
                    "SAM2 model checkpoint not found.\n"
                    "Please download the model and place it in the checkpoints folder."
                )
                return
                
            # Create enhanced predictor
            self.predictor = EnhancedSAM2Predictor(
                model_cfg,
                checkpoint_path,
                torch.device(self.settings['device']),
                self.settings
            )
            
            # Load model in background
            self.status_label.configure(text="Loading SAM2 model...")
            threading.Thread(target=self._load_model_thread, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            messagebox.showerror("Error", f"Failed to initialize SAM2: {str(e)}")
            
    def _load_model_thread(self):
        """Load model in background thread"""
        try:
            self.predictor.load_model()
            self.root.after(0, lambda: self.status_label.configure(text="Model loaded successfully"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {str(e)}"))
            
    def process_video(self):
        """Start video processing"""
        video_path = self.video_path_var.get()
        
        if not video_path or not os.path.exists(video_path):
            messagebox.showwarning("Warning", "Please select a valid video file")
            return
            
        if not self.predictor or not self.predictor.model_loaded:
            messagebox.showwarning("Warning", "Model is still loading. Please wait...")
            return
            
        # Create project directory
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_dir = os.path.join("projects", f"{video_name}_{timestamp}")
        os.makedirs(project_dir, exist_ok=True)
        
        # Save project info
        self.current_project = {
            'video_path': video_path,
            'project_dir': project_dir,
            'settings': {
                'skip_frames': self.skip_frames_var.get(),
                'resize_factor': self.resize_var.get(),
                'overlap_threshold': self.threshold_var.get() / 100.0,
                'export_video': self.export_video_var.get(),
                'export_csv': self.export_csv_var.get(),
                'export_elan': self.export_elan_var.get()
            }
        }
        
        # Save project metadata
        with open(os.path.join(project_dir, 'project.json'), 'w') as f:
            json.dump(self.current_project, f, indent=2)
            
        # Start processing in thread
        self.processing_thread = threading.Thread(
            target=self._process_video_thread,
            daemon=True
        )
        self.processing_thread.start()
        
        # Update UI
        self.process_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.notebook.select(self.progress_frame)
        
    def _process_video_thread(self):
        """Process video in background thread"""
        try:
            project = self.current_project
            project_dir = project['project_dir']
            settings = project['settings']
            
            # Step 1: Extract frames
            self.update_progress("Extracting frames...")
            processor = EnhancedVideoProcessor(
                project['video_path'],
                project_dir,
                self.settings
            )
            
            fps, num_frames = processor.extract_frames_optimized(
                skip_frames=settings['skip_frames'],
                resize_factor=settings['resize_factor']
            )
            
            if fps == -1:
                raise Exception("Failed to extract frames")
                
            self.update_progress(f"Extracted {num_frames} frames at {fps:.1f} FPS")
            
            # Step 2: Select reference frame and annotate
            frames_dir = os.path.join(project_dir, "frames")
            reference_frame_idx = num_frames // 2
            
            # Load reference frame
            ref_frame_path = os.path.join(frames_dir, f"{reference_frame_idx:05d}.jpg")
            ref_frame = cv2.imread(ref_frame_path)
            
            # Get annotations (run in main thread)
            self.root.after(0, lambda: self._get_annotations(ref_frame, reference_frame_idx))
            
            # Wait for annotations
            while not hasattr(self, 'current_annotations'):
                time.sleep(0.1)
                
            if self.current_annotations is None:
                raise Exception("Annotation cancelled")
                
            points_dict, labels_dict, object_names = self.current_annotations
            delattr(self, 'current_annotations')
            
            # Step 3: Process with SAM2
            self.update_progress("Processing with SAM2...")
            
            # Prepare prompts
            prompts = {}
            for obj_id in points_dict:
                prompts[obj_id] = {
                    'points': points_dict[obj_id],
                    'labels': labels_dict[obj_id],
                    'frame_idx': reference_frame_idx,
                    'name': object_names.get(obj_id, f"Object_{obj_id}")
                }
                
            # Process video
            results = {}
            frame_count = 0
            
            def progress_callback(frame_idx, frame_results):
                nonlocal frame_count
                frame_count += 1
                if frame_count % 10 == 0:
                    self.update_progress(f"Processing frame {frame_count}/{num_frames}")
                    
            for frame_idx, frame_results in self.predictor.process_video_streaming(
                frames_dir, prompts, progress_callback
            ):
                results[frame_idx] = frame_results
                
            self.update_progress(f"Processed {len(results)} frames")
            
            # Step 4: Analyze overlaps
            if settings['overlap_threshold'] > 0:
                self.update_progress("Analyzing overlaps...")
                overlap_analyzer = EnhancedOverlapAnalyzer(
                    settings['overlap_threshold'],
                    object_names
                )
                
                overlap_results = overlap_analyzer.analyze_video(results)
                
                # Save overlap analysis
                overlap_path = os.path.join(project_dir, "overlap_analysis.json")
                with open(overlap_path, 'w') as f:
                    json.dump(overlap_results, f, indent=2)
                    
            # Step 5: Export results
            self.update_progress("Exporting results...")
            
            # Export video
            if settings['export_video']:
                self.update_progress("Creating annotated video...")
                output_video_path = os.path.join(project_dir, "annotated_video.mp4")
                self._export_video(
                    frames_dir,
                    results,
                    output_video_path,
                    fps,
                    object_names,
                    overlap_results if settings['overlap_threshold'] > 0 else None
                )
                
            # Export CSV
            if settings['export_csv']:
                self.update_progress("Exporting CSV data...")
                self._export_csv(
                    results,
                    os.path.join(project_dir, "tracking_data.csv"),
                    fps,
                    object_names
                )
                
            # Export ELAN
            if settings['export_elan'] and settings['overlap_threshold'] > 0:
                self.update_progress("Creating ELAN file...")
                self._export_elan(
                    overlap_results,
                    os.path.join(project_dir, "annotations.eaf"),
                    project['video_path'],
                    fps
                )
                
            # Done
            self.update_progress("Processing completed successfully!")
            
            # Show results
            self.root.after(0, lambda: self._show_results(project_dir))
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.update_progress(f"Error: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Processing Failed", str(e)))
            
        finally:
            # Reset UI
            self.root.after(0, self._reset_ui)
            
    def _get_annotations(self, frame, reference_frame_idx):
        """Get annotations from user"""
        try:
            # Create annotation dialog
            dialog = AnnotationDialog(self.root, frame, reference_frame_idx)
            self.root.wait_window(dialog.dialog)
            
            if dialog.result:
                self.current_annotations = dialog.result
            else:
                self.current_annotations = None
                
        except Exception as e:
            logger.error(f"Annotation failed: {e}")
            self.current_annotations = None
            
    def _export_video(self, frames_dir, results, output_path, fps, object_names, overlap_results=None):
        """Export annotated video"""
        # Get frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        
        if not frame_files:
            return
            
        # Get video dimensions
        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        height, width = first_frame.shape[:2]
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        # Color map
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        # Process frames
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                continue
                
            # Create overlay
            overlay = frame.copy()
            
            # Apply masks
            if i in results:
                for obj_id, mask in results[i].items():
                    if mask.shape[:2] != (height, width):
                        mask = cv2.resize(mask.astype(np.float32), (width, height)) > 0.5
                        
                    # Get color
                    color = (colors[obj_id % 10][:3] * 255).astype(np.uint8)
                    
                    # Apply mask
                    overlay[mask] = overlay[mask] * 0.5 + color * 0.5
                    
                    # Add label
                    obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                    moments = cv2.moments(mask.astype(np.uint8))
                    if moments['m00'] > 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                        
                        cv2.putText(overlay, obj_name, (cx - 30, cy),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
            # Add overlap info if available
            if overlap_results and i in overlap_results.get('frame_events', {}):
                events = overlap_results['frame_events'][i]
                y_pos = 30
                for event in events:
                    text = f"{event['source']} -> {event['target']}"
                    cv2.putText(overlay, text, (10, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_pos += 25
                    
            # Combine frames
            combined = np.hstack([frame, overlay])
            out.write(combined)
            
        out.release()
        self.update_progress(f"Video saved: {output_path}")
        
    def _export_csv(self, results, output_path, fps, object_names):
        """Export tracking data to CSV"""
        data = []
        
        for frame_idx in sorted(results.keys()):
            frame_data = {
                'frame': frame_idx,
                'time': frame_idx / fps
            }
            
            for obj_id, mask in results[frame_idx].items():
                obj_name = object_names.get(obj_id, f"Object_{obj_id}")
                
                # Calculate centroid and area
                moments = cv2.moments(mask.astype(np.uint8))
                if moments['m00'] > 0:
                    cx = moments['m10'] / moments['m00']
                    cy = moments['m01'] / moments['m00']
                    area = moments['m00']
                else:
                    cx = cy = area = 0
                    
                frame_data[f'{obj_name}_x'] = cx
                frame_data[f'{obj_name}_y'] = cy
                frame_data[f'{obj_name}_area'] = area
                
            data.append(frame_data)
            
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        self.update_progress(f"CSV saved: {output_path}")
        
    def _export_elan(self, overlap_results, output_path, video_path, fps):
        """Export ELAN annotation file"""
        # Create ELAN XML structure
        root = ET.Element("ANNOTATION_DOCUMENT")
        root.set("AUTHOR", "SAM2_Enhanced")
        root.set("DATE", datetime.now().isoformat())
        root.set("FORMAT", "3.0")
        root.set("VERSION", "3.0")
        
        # Add header
        header = ET.SubElement(root, "HEADER")
        header.set("MEDIA_FILE", "")
        header.set("TIME_UNITS", "milliseconds")
        
        # Add media descriptor
        media = ET.SubElement(header, "MEDIA_DESCRIPTOR")
        media.set("MEDIA_URL", f"file://{os.path.abspath(video_path)}")
        media.set("MIME_TYPE", "video/mp4")
        
        # Add time order
        time_order = ET.SubElement(root, "TIME_ORDER")
        
        # Create time slots for all events
        time_slot_id = 1
        time_slots = {}
        
        for event in overlap_results.get('events', []):
            start_ms = int(event['start_time'] * 1000)
            end_ms = int(event['end_time'] * 1000)
            
            if start_ms not in time_slots:
                ts = ET.SubElement(time_order, "TIME_SLOT")
                ts.set("TIME_SLOT_ID", f"ts{time_slot_id}")
                ts.set("TIME_VALUE", str(start_ms))
                time_slots[start_ms] = f"ts{time_slot_id}"
                time_slot_id += 1
                
            if end_ms not in time_slots:
                ts = ET.SubElement(time_order, "TIME_SLOT")
                ts.set("TIME_SLOT_ID", f"ts{time_slot_id}")
                ts.set("TIME_VALUE", str(end_ms))
                time_slots[end_ms] = f"ts{time_slot_id}"
                time_slot_id += 1
                
        # Create tiers for each object
        annotation_id = 1
        
        for obj_name in overlap_results.get('objects', []):
            tier = ET.SubElement(root, "TIER")
            tier.set("TIER_ID", obj_name.replace(' ', '_'))
            tier.set("LINGUISTIC_TYPE_REF", "default")
            
            # Add annotations for this object
            for event in overlap_results.get('events', []):
                if event['source'] == obj_name:
                    annotation = ET.SubElement(tier, "ANNOTATION")
                    alignable = ET.SubElement(annotation, "ALIGNABLE_ANNOTATION")
                    alignable.set("ANNOTATION_ID", f"a{annotation_id}")
                    alignable.set("TIME_SLOT_REF1", time_slots[int(event['start_time'] * 1000)])
                    alignable.set("TIME_SLOT_REF2", time_slots[int(event['end_time'] * 1000)])
                    
                    value = ET.SubElement(alignable, "ANNOTATION_VALUE")
                    value.text = f"Looking at {event['target']}"
                    
                    annotation_id += 1
                    
        # Add linguistic type
        ling_type = ET.SubElement(root, "LINGUISTIC_TYPE")
        ling_type.set("LINGUISTIC_TYPE_ID", "default")
        ling_type.set("TIME_ALIGNABLE", "true")
        
        # Save file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='UTF-8', xml_declaration=True)
        self.update_progress(f"ELAN file saved: {output_path}")
        
    def _show_results(self, project_dir):
        """Show results in UI"""
        self.notebook.select(self.results_frame)
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        # Create results display
        results_text = ctk.CTkTextbox(self.results_frame)
        results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add results summary
        results_text.insert("end", "Processing Completed Successfully!\n\n")
        results_text.insert("end", f"Project Directory: {project_dir}\n\n")
        
        # List generated files
        results_text.insert("end", "Generated Files:\n")
        for file in os.listdir(project_dir):
            file_path = os.path.join(project_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                results_text.insert("end", f"  • {file} ({size:.1f} MB)\n")
                
        # Add open folder button
        open_btn = ctk.CTkButton(
            self.results_frame,
            text="Open Project Folder",
            command=lambda: os.startfile(project_dir) if sys.platform == "win32" else subprocess.call(["open", project_dir])
        )
        open_btn.pack(pady=10)
        
    def _reset_ui(self):
        """Reset UI after processing"""
        self.process_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.processing_thread = None
        
    def update_progress(self, message):
        """Update progress display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.root.after(0, lambda: self._update_progress_ui(f"[{timestamp}] {message}\n"))
        
    def _update_progress_ui(self, message):
        """Update progress UI in main thread"""
        self.progress_text.insert("end", message)
        self.progress_text.see("end")
        self.status_label.configure(text=message.strip())
        
    def stop_processing(self):
        """Stop current processing"""
        if self.processing_thread and self.processing_thread.is_alive():
            # This is a simplified stop - in production you'd want proper thread interruption
            messagebox.showinfo("Stop", "Processing will stop after current frame")
            
    def run(self):
        """Run the application"""
        self.root.mainloop()


# ================== ENHANCED OVERLAP ANALYZER ==================

class EnhancedOverlapAnalyzer:
    """Enhanced overlap analysis with better algorithms"""
    
    def __init__(self, threshold, object_names):
        self.threshold = threshold
        self.object_names = object_names
        
    def analyze_video(self, results):
        """Analyze overlaps throughout video"""
        analysis = {
            'objects': list(self.object_names.values()),
            'events': [],
            'frame_events': {},
            'statistics': {}
        }
        
        # Track ongoing events
        active_events = {}
        
        for frame_idx in sorted(results.keys()):
            frame_overlaps = self.analyze_frame(results[frame_idx])
            
            # Store frame events
            if frame_overlaps:
                analysis['frame_events'][frame_idx] = frame_overlaps
                
            # Update active events
            current_pairs = {(e['source'], e['target']) for e in frame_overlaps}
            
            # End events that are no longer active
            ended_events = []
            for pair, event in active_events.items():
                if pair not in current_pairs:
                    event['end_frame'] = frame_idx - 1
                    event['duration'] = event['end_frame'] - event['start_frame'] + 1
                    analysis['events'].append(event)
                    ended_events.append(pair)
                    
            for pair in ended_events:
                del active_events[pair]
                
            # Start new events
            for overlap in frame_overlaps:
                pair = (overlap['source'], overlap['target'])
                if pair not in active_events:
                    active_events[pair] = {
                        'source': overlap['source'],
                        'target': overlap['target'],
                        'start_frame': frame_idx,
                        'start_time': frame_idx / 30.0,  # Will be corrected with actual FPS
                        'type': overlap['type']
                    }
                    
        # Close any remaining events
        for event in active_events.values():
            event['end_frame'] = max(results.keys())
            event['duration'] = event['end_frame'] - event['start_frame'] + 1
            analysis['events'].append(event)
            
        # Calculate statistics
        analysis['statistics'] = self.calculate_statistics(analysis['events'])
        
        return analysis
        
    def analyze_frame(self, frame_results):
        """Analyze overlaps in a single frame"""
        overlaps = []
        
        obj_ids = list(frame_results.keys())
        
        for i, obj1_id in enumerate(obj_ids):
            for obj2_id in obj_ids[i+1:]:
                mask1 = frame_results[obj1_id]
                mask2 = frame_results[obj2_id]
                
                overlap_info = self.calculate_overlap(mask1, mask2)
                
                if overlap_info and overlap_info['percentage'] >= self.threshold:
                    overlaps.append({
                        'source': self.object_names.get(obj1_id, f"Object_{obj1_id}"),
                        'target': self.object_names.get(obj2_id, f"Object_{obj2_id}"),
                        'percentage': overlap_info['percentage'],
                        'type': overlap_info['type']
                    })
                    
        return overlaps
        
    def calculate_overlap(self, mask1, mask2):
        """Calculate overlap between two masks"""
        if mask1.shape != mask2.shape:
            return None
            
        # Ensure binary masks
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)
        
        # Calculate areas
        area1 = np.sum(mask1)
        area2 = np.sum(mask2)
        
        if area1 == 0 or area2 == 0:
            return None
            
        # Calculate intersection
        intersection = np.sum(mask1 & mask2)
        
        if intersection == 0:
            return None
            
        # Calculate overlap percentage
        overlap_pct = intersection / min(area1, area2) * 100
        
        # Determine overlap type
        if overlap_pct > 80:
            overlap_type = "inclusion"
        elif overlap_pct > 20:
            overlap_type = "partial"
        else:
            overlap_type = "touch"
            
        return {
            'percentage': overlap_pct,
            'type': overlap_type,
            'intersection': intersection,
            'area1': area1,
            'area2': area2
        }
        
    def calculate_statistics(self, events):
        """Calculate event statistics"""
        if not events:
            return {}
            
        stats = {
            'total_events': len(events),
            'avg_duration': np.mean([e['duration'] for e in events]),
            'max_duration': max(e['duration'] for e in events),
            'min_duration': min(e['duration'] for e in events)
        }
        
        # Count by type
        type_counts = {}
        for event in events:
            event_type = event.get('type', 'unknown')
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            
        stats['type_distribution'] = type_counts
        
        return stats


# ================== ANNOTATION DIALOG ==================

class AnnotationDialog:
    """Modern annotation dialog for point selection"""
    
    def __init__(self, parent, frame, frame_idx):
        self.parent = parent
        self.frame = frame.copy()
        self.frame_idx = frame_idx
        self.result = None
        
        # Data storage
        self.points_dict = {}
        self.labels_dict = {}
        self.object_names = {}
        self.current_obj_id = 1
        
        # Create dialog
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title(f"Annotate Frame {frame_idx}")
        self.dialog.geometry("1000x700")
        
        self.setup_ui()
        self.bind_events()
        
    def setup_ui(self):
        """Setup annotation UI"""
        # Main container
        main_frame = ctk.CTkFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Canvas
        self.canvas_frame = ctk.CTkFrame(main_frame)
        self.canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Canvas for image
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)
        
        # Right panel - Controls
        self.control_frame = ctk.CTkFrame(main_frame, width=250)
        self.control_frame.pack(side="right", fill="y")
        self.control_frame.pack_propagate(False)
        
        # Instructions
        instructions = ctk.CTkLabel(
            self.control_frame,
            text="Click Selection\n\nLeft Click: Positive point\nRight Click: Negative point",
            font=ctk.CTkFont(size=14)
        )
        instructions.pack(pady=20)
        
        # Current object
        self.obj_label = ctk.CTkLabel(
            self.control_frame,
            text=f"Current Object: {self.current_obj_id}",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.obj_label.pack(pady=10)
        
        # Object name entry
        name_frame = ctk.CTkFrame(self.control_frame)
        name_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(name_frame, text="Name:").pack(anchor="w")
        self.name_entry = ctk.CTkEntry(name_frame)
        self.name_entry.pack(fill="x", pady=5)
        self.name_entry.insert(0, f"Object_{self.current_obj_id}")
        
        # Buttons
        btn_frame = ctk.CTkFrame(self.control_frame)
        btn_frame.pack(fill="x", padx=20, pady=20)
        
        self.next_btn = ctk.CTkButton(
            btn_frame,
            text="Next Object",
            command=self.next_object
        )
        self.next_btn.pack(fill="x", pady=5)
        
        self.prev_btn = ctk.CTkButton(
            btn_frame,
            text="Previous Object",
            command=self.prev_object
        )
        self.prev_btn.pack(fill="x", pady=5)
        
        self.clear_btn = ctk.CTkButton(
            btn_frame,
            text="Clear Points",
            command=self.clear_points
        )
        self.clear_btn.pack(fill="x", pady=5)
        
        # Target checkbox
        self.is_target_var = tk.BooleanVar()
        self.target_cb = ctk.CTkCheckBox(
            self.control_frame,
            text="Mark as Target",
            variable=self.is_target_var
        )
        self.target_cb.pack(pady=10)
        
        # Done button
        self.done_btn = ctk.CTkButton(
            self.control_frame,
            text="Done",
            command=self.done,
            height=40,
            fg_color="green"
        )
        self.done_btn.pack(side="bottom", fill="x", padx=20, pady=20)
        
        # Display image
        self.display_image()
        
    def display_image(self):
        """Display the frame with annotations"""
        # Convert frame to RGB
        display_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        
        # Draw existing points
        for obj_id in self.points_dict:
            for point, label in zip(self.points_dict[obj_id], self.labels_dict[obj_id]):
                color = (0, 255, 0) if label == 1 else (255, 0, 0)
                cv2.circle(display_frame, (int(point[0]), int(point[1])), 5, color, -1)
                
                # Add label
                obj_name = self.object_names.get(obj_id, f"Object_{obj_id}")
                cv2.putText(display_frame, f"{obj_id}", 
                          (int(point[0] + 10), int(point[1] - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # Convert to PhotoImage
        image = Image.fromarray(display_frame)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
        self.photo = ImageTk.PhotoImage(image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas_image = self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            anchor="center",
            image=self.photo
        )
        
        # Store scaling factors
        self.scale_x = self.frame.shape[1] / image.width
        self.scale_y = self.frame.shape[0] / image.height
        self.offset_x = (canvas_width - image.width) // 2
        self.offset_y = (canvas_height - image.height) // 2
        
    def bind_events(self):
        """Bind mouse events"""
        self.canvas.bind("<Button-1>", lambda e: self.add_point(e, 1))
        self.canvas.bind("<Button-3>", lambda e: self.add_point(e, 0))
        self.canvas.bind("<Configure>", lambda e: self.display_image())
        
    def add_point(self, event, label):
        """Add a point annotation"""
        # Convert canvas coordinates to image coordinates
        x = (event.x - self.offset_x) * self.scale_x
        y = (event.y - self.offset_y) * self.scale_y
        
        # Ensure within bounds
        x = max(0, min(x, self.frame.shape[1] - 1))