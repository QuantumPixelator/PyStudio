from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QLineEdit, QTextEdit, QPushButton, QProgressBar, QGroupBox, QScrollBar, QMenu, QCheckBox,
    QFileDialog, QMessageBox, QSlider)
from PySide6.QtGui import QIcon, QAction, QContextMenuEvent, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, Slot
import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import random

MODELS_DIR = "./models"

class ModelLoaderThread(QThread):
    model_loaded = Signal(object)
    log = Signal(str)
    def __init__(self, model_name, available_models, current_model, pipe):
        super().__init__()
        self.model_name = model_name
        self.available_models = available_models
        self.current_model = current_model
        self.pipe = pipe
    def run(self):
        # ...model loading logic (ported from Tkinter version)...
        self.model_loaded.emit(self.pipe)

class SDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyStudio Diffusion")
        self.setGeometry(100, 100, 700, 750)
        self.setFixedSize(700, 750)

        # Model and image generation state
        self.pipe = None
        self.current_model = None
        self.is_generating = False
        self._cancel_requested = False

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Model & Image Settings
        model_group = QGroupBox("Model & Image Settings")
        model_layout = QGridLayout(model_group)
        main_layout.addWidget(model_group)

        model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        model_layout.addWidget(model_label, 0, 0)
        model_layout.addWidget(self.model_combo, 0, 1)

        size_label = QLabel("Image Size:")
        self.size_combo = QComboBox()
        self.size_options = ["512x512", "768x768", "1024x1024", "1920x1080"]
        self.size_combo.addItems(self.size_options)
        model_layout.addWidget(size_label, 0, 2)
        model_layout.addWidget(self.size_combo, 0, 3)

        lora_label = QLabel("Select LoRA:")
        self.lora_combo = QComboBox()
        model_layout.addWidget(lora_label, 1, 0)
        model_layout.addWidget(self.lora_combo, 1, 1)

        # Generation Options
        options_group = QGroupBox("Generation Options")
        options_layout = QGridLayout(options_group)
        main_layout.addWidget(options_group)

        guidance_label = QLabel("Prompt Adherence / Creativity:")
        self.guidance_slider = QSlider(Qt.Horizontal)
        self.guidance_slider.setMinimum(1)
        self.guidance_slider.setMaximum(20)
        self.guidance_slider.setValue(7)
        self.guidance_slider.setTickInterval(1)
        self.guidance_slider.setTickPosition(QSlider.TicksBelow)
        self.guidance_value_label = QLabel("7.0")
        self.guidance_slider.valueChanged.connect(lambda v: self.guidance_value_label.setText(f"{v:.2f}"))
        options_layout.addWidget(guidance_label, 0, 0)
        options_layout.addWidget(self.guidance_slider, 0, 1)
        options_layout.addWidget(self.guidance_value_label, 0, 2)

        steps_label = QLabel("Quality / Detail (Steps):")
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setMinimum(10)
        self.steps_slider.setMaximum(100)
        self.steps_slider.setValue(30)
        self.steps_slider.setTickInterval(1)
        self.steps_slider.setTickPosition(QSlider.TicksBelow)
        self.steps_value_label = QLabel("30")
        self.steps_slider.valueChanged.connect(lambda v: self.steps_value_label.setText(str(v)))
        options_layout.addWidget(steps_label, 1, 0)
        options_layout.addWidget(self.steps_slider, 1, 1)
        options_layout.addWidget(self.steps_value_label, 1, 2)

        cfg_label = QLabel("CFG (LoRA Strength):")
        self.cfg_slider = QSlider(Qt.Horizontal)
        self.cfg_slider.setMinimum(0)
        self.cfg_slider.setMaximum(100)
        self.cfg_slider.setValue(70)
        self.cfg_slider.setTickInterval(1)
        self.cfg_slider.setTickPosition(QSlider.TicksBelow)
        self.cfg_value_label = QLabel("0.70")
        self.cfg_slider.valueChanged.connect(lambda v: self.cfg_value_label.setText(f"{v/100:.2f}"))
        options_layout.addWidget(cfg_label, 2, 0)
        options_layout.addWidget(self.cfg_slider, 2, 1)
        options_layout.addWidget(self.cfg_value_label, 2, 2)

        seed_label = QLabel("Random Seed:")
        self.seed_entry = QLineEdit()
        options_layout.addWidget(seed_label, 3, 0)
        options_layout.addWidget(self.seed_entry, 3, 1)
        self.auto_clear_seed_check = QCheckBox("Auto Clear Seed")
        options_layout.addWidget(self.auto_clear_seed_check, 3, 2)

        # Prompts
        prompts_layout = QHBoxLayout()
        main_layout.addLayout(prompts_layout)

        prompt_group = QGroupBox("Image Description (Prompt)")
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_entry = QTextEdit()
        prompt_layout.addWidget(self.prompt_entry)
        prompts_layout.addWidget(prompt_group)

        negative_group = QGroupBox("Negative Prompt (optional)")
        negative_layout = QVBoxLayout(negative_group)
        self.negative_entry = QTextEdit()
        negative_layout.addWidget(self.negative_entry)
        prompts_layout.addWidget(negative_group)

        # Generate/Cancel buttons
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        self.generate_button = QPushButton("Generate Image")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.cancel_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready to generate an image")
        main_layout.addWidget(self.status_label)

        # Log box
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)

        # Model and LoRA population
        self.available_models = self._find_local_models()
        self.model_combo.addItems(list(self.available_models.keys()))
        self.lora_options = self._find_loras()
        self.lora_combo.addItems(self.lora_options)

        # Log app startup
        self.log("App started. Ready to generate an image.")

        # Auto-load negative prompt from negative.txt if it exists
        neg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "negative.txt")
        if os.path.isfile(neg_path):
            try:
                with open(neg_path, "r", encoding="utf-8") as nf:
                    neg_text = nf.read().strip()
                self.negative_entry.setPlainText(neg_text)
            except Exception:
                pass

        # Right-click context menu for prompt/negative fields
        self.prompt_entry.setContextMenuPolicy(Qt.CustomContextMenu)
        self.negative_entry.setContextMenuPolicy(Qt.CustomContextMenu)
        self.prompt_entry.customContextMenuRequested.connect(lambda pos: self._show_text_menu(self.prompt_entry, pos))
        self.negative_entry.customContextMenuRequested.connect(lambda pos: self._show_text_menu(self.negative_entry, pos))

        # Model and image generation state
        self.pipe = None
        self.current_model = None
        self.is_generating = False
        self._cancel_requested = False

        self.generation_thread = None
        self.generate_button.clicked.connect(self.generate_image)
        self.cancel_button.clicked.connect(self.cancel_generation)

    def cancel_generation(self):
        self._cancel_requested = True
        self.log("Image generation cancelled by user.")
        self.status_label.setText("Generation cancelled.")
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        # Terminate the generation thread if running
        if self.generation_thread and self.generation_thread.isRunning():
            self.generation_thread.terminate()
            self.generation_thread.wait()
            self.generation_thread = None

    def load_model_threaded(self, callback=None):
        def load():
            model_name = self.model_combo.currentText()
            model_dir = self.available_models.get(model_name, None)
            if not model_name or not model_dir:
                self.log("No model selected or model not found.")
                if callback:
                    callback(None)
                return
            if self.current_model == model_name and self.pipe is not None:
                if callback:
                    callback(self.pipe)
                return
            self.log(f"Loading model: {model_name}")
            QApplication.processEvents()
            try:
                use_cuda = torch.cuda.is_available()
                if not use_cuda:
                    self.log("CUDA GPU not detected. Image generation will be much slower and may produce poor results, especially for SDXL.")
                    QMessageBox.warning(self, "CUDA Not Available", "CUDA GPU not detected. Image generation will be much slower and may produce poor results, especially for SDXL.")
                torch_dtype = torch.float16 if use_cuda else torch.float32
                files = os.listdir(model_dir)
                has_index = "model_index.json" in files
                ckpt_files = [f for f in files if f.endswith(".ckpt") or f.endswith(".safetensors")]
                if has_index:
                    if "xl" in model_name.lower():
                        pipe = StableDiffusionXLPipeline.from_pretrained(model_dir, torch_dtype=torch_dtype, safety_checker=None)
                    else:
                        pipe = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch_dtype, safety_checker=None)
                elif ckpt_files:
                    ckpt_path = os.path.join(model_dir, ckpt_files[0])
                    try:
                        if "xl" in model_name.lower() and hasattr(StableDiffusionXLPipeline, "from_single_file"):
                            pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, torch_dtype=torch_dtype, safety_checker=None)
                        elif hasattr(StableDiffusionPipeline, "from_single_file"):
                            pipe = StableDiffusionPipeline.from_single_file(ckpt_path, torch_dtype=torch_dtype, safety_checker=None)
                        else:
                            raise RuntimeError("Your diffusers version does not support from_single_file. Please update diffusers.")
                    except Exception as e:
                        self.log(f"Failed to load single-file model: {e}")
                        QMessageBox.critical(self, "Error", f"Failed to load single-file model: {e}")
                        self.status_label.setText("Model loading failed")
                        if callback:
                            callback(None)
                        return
                else:
                    raise RuntimeError("No supported model files found in folder.")
                device = "cuda" if use_cuda else "cpu"
                pipe = pipe.to(device)
                self.pipe = pipe
                self.current_model = model_name
                self.log(f"Loaded model: {model_name} (Device: {device})")
                if callback:
                    callback(pipe)
            except Exception as e:
                self.log(f"Failed to load model: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
                self.status_label.setText("Model loading failed")
                if callback:
                    callback(None)
        thread = QThread()
        thread.run = load
        thread.start()

    def generate_image(self):
        if self.is_generating:
            self.log("Image generation in progress! Please wait.")
            QMessageBox.warning(self, "Warning", "Image generation in progress! Please wait.")
            return
        prompt = self.prompt_entry.toPlainText().strip()
        negative_prompt = self.negative_entry.toPlainText().strip()
        model_name = self.model_combo.currentText()
        token_limit = 77
        if model_name and "xl" in model_name.lower():
            token_limit = 120
        tokenizer = None
        if self.pipe and hasattr(self.pipe, "tokenizer"):
            tokenizer = self.pipe.tokenizer
        elif self.pipe and hasattr(self.pipe, "text_encoder") and hasattr(self.pipe.text_encoder, "tokenizer"):
            tokenizer = self.pipe.text_encoder.tokenizer
        try:
            if tokenizer:
                tokens = tokenizer(prompt, truncation=False, return_tensors=None)["input_ids"]
                token_count = len(tokens)
            else:
                token_count = len(prompt.split())
        except Exception:
            token_count = len(prompt.split())
        if token_count > token_limit:
            self.log(f"Prompt token count: {token_count} (limit: {token_limit}) - Prompt will be truncated!")
            QMessageBox.warning(self, "Prompt Truncated", f"Your prompt is {token_count} tokens (limit: {token_limit}). It will be truncated.")
        else:
            self.log(f"Prompt token count: {token_count} (limit: {token_limit})")
        if not model_name or model_name not in self.available_models:
            self.log("Please select a model before generating an image.")
            QMessageBox.warning(self, "Model Error", "Please select a model before generating an image.")
            return
        if not prompt:
            self.log("Please enter a description!")
            QMessageBox.warning(self, "Input Error", "Please enter a description!")
            return
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.is_generating = True
        self._cancel_requested = False
        if self.auto_clear_seed_check.isChecked():
            self.seed_entry.clear()
        self.log("Loading model...")
        self.progress_bar.setValue(0)
        def after_model_loaded(pipe):
            if pipe is None:
                self.is_generating = False
                self.generate_button.setEnabled(True)
                self.cancel_button.setEnabled(False)
                return
            self.log("Generating image...")
            def run_generation():
                def progress_callback(step, timestep, total_steps):
                    percent = int((step + 1) / total_steps * 100)
                    self.progress_bar.setValue(percent)
                    QApplication.processEvents()
                try:
                    size_str = self.size_combo.currentText()
                    if "x" in size_str:
                        width, height = map(int, size_str.split("x"))
                    else:
                        width, height = 512, 512
                    guidance_scale = float(self.guidance_slider.value())
                    num_inference_steps = int(self.steps_slider.value())
                    seed_str = self.seed_entry.text().strip()
                    generator = None
                    used_seed = None
                    if seed_str:
                        try:
                            seed = int(seed_str)
                            generator = torch.manual_seed(seed)
                            used_seed = seed
                        except ValueError:
                            QMessageBox.warning(self, "Seed Error", "Seed must be an integer or blank.")
                            generator = None
                    else:
                        used_seed = random.randint(0, 2**32 - 1)
                        generator = torch.manual_seed(used_seed)
                    use_cuda = torch.cuda.is_available()
                    lora_name = self.lora_combo.currentText()
                    lora_path = None
                    if lora_name:
                        loras_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loras")
                        lora_path = os.path.join(loras_dir, lora_name)
                        if os.path.isdir(lora_path):
                            try:
                                safetensors_files = [f for f in os.listdir(lora_path) if f.endswith(".safetensors")]
                                if safetensors_files:
                                    lora_weights = os.path.join(lora_path, safetensors_files[0])
                                    if hasattr(pipe, "load_lora_weights"):
                                        pipe.load_lora_weights(lora_weights, weight=self.cfg_slider.value()/100)
                                else:
                                    ckpt_files = [f for f in os.listdir(lora_path) if f.endswith(".ckpt")]
                                    if ckpt_files:
                                        lora_weights = os.path.join(lora_path, ckpt_files[0])
                                        if hasattr(pipe, "load_lora_weights"):
                                            pipe.load_lora_weights(lora_weights, weight=self.cfg_slider.value()/100)
                            except Exception as e:
                                QMessageBox.critical(self, "LoRA Compatibility Error", f"LoRA compatibility error! Generation stopped. Please select a compatible model or LoRA.")
                                self.log(f"LoRA compatibility error! Generation stopped. Please select a compatible model or LoRA.")
                                self.is_generating = False
                                self.generate_button.setEnabled(True)
                                self.cancel_button.setEnabled(False)
                                self.progress_bar.setValue(0)
                                return
                    with torch.autocast("cuda" if use_cuda else "cpu"):
                        pipe_args = dict(
                            prompt=prompt,
                            negative_prompt=negative_prompt if negative_prompt else None,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            height=height,
                            width=width,
                            generator=generator,
                            callback_on_step_end=progress_callback
                        )
                        try:
                            result = pipe(**pipe_args)
                        except TypeError:
                            pipe_args.pop('callback_on_step_end', None)
                            try:
                                result = pipe(**pipe_args)
                            except Exception:
                                self.log("Progress bar not supported in this diffusers/model version.")
                                pipe_args = dict(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt if negative_prompt else None,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    height=height,
                                    width=width,
                                    generator=generator
                                )
                                result = pipe(**pipe_args)
                        image = result.images[0]
                    if self._cancel_requested:
                        self.log("Generation cancelled before completion.")
                        self.status_label.setText("Generation cancelled.")
                        self.is_generating = False
                        self.generate_button.setEnabled(True)
                        self.cancel_button.setEnabled(False)
                        self.progress_bar.setValue(0)
                        return
                    if hasattr(image, 'getextrema'):
                        extrema = image.getextrema()
                        if isinstance(extrema, tuple) and all(e == (0, 0) for e in extrema):
                            self.log("Generated image is completely black. This usually means a model or device issue. Try restarting, updating diffusers, or verifying your model files.")
                            QMessageBox.warning(self, "Black Image Warning", "Generated image is completely black. This usually means a model or device issue. Try restarting, updating diffusers, or verifying your model files.")
                    self.seed_entry.setText(str(used_seed))
                    self.log("Image generated. Right-click the image to save.")
                    self.progress_bar.setValue(100)
                    self.show_image_window(image)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to generate image: {e}")
                    self.log(f"Image generation failed: {e}")
                    self.progress_bar.setValue(0)
                finally:
                    self.is_generating = False
                    self.generate_button.setEnabled(True)
                    self.cancel_button.setEnabled(False)
            self.generation_thread = QThread()
            self.generation_thread.run = run_generation
            self.generation_thread.start()
        self.load_model_threaded(callback=after_model_loaded)

    def show_image_window(self, image):
        win = QWidget()
        win.setWindowTitle("Generated Image")
        layout = QVBoxLayout(win)
        img_disp = image.copy()
        img_disp.thumbnail((800, 800), Image.Resampling.LANCZOS)
        img_path = "_temp_img.png"
        img_disp.save(img_path)
        pixmap = QPixmap(img_path)
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        layout.addWidget(img_label)
        def save_image():
            filename, _ = QFileDialog.getSaveFileName(win, "Save Generated Image", "", "PNG files (*.png);;All files (*)")
            if filename:
                image.save(filename)
                self.log(f"Image saved as {os.path.basename(filename)}")
            else:
                self.log("Image save cancelled")
        img_label.setContextMenuPolicy(Qt.CustomContextMenu)
        def show_img_menu(pos):
            menu = QMenu()
            menu.addAction("Save Image", save_image)
            menu.exec(img_label.mapToGlobal(pos))
        img_label.customContextMenuRequested.connect(show_img_menu)
        win.show()
        # Auto-load negative prompt from negative.txt if it exists
        neg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "negative.txt")
        if os.path.isfile(neg_path):
            try:
                with open(neg_path, "r", encoding="utf-8") as nf:
                    neg_text = nf.read().strip()
                self.negative_entry.setPlainText(neg_text)
            except Exception:
                pass

        # Right-click context menu for prompt/negative fields
        self.prompt_entry.setContextMenuPolicy(Qt.CustomContextMenu)
        self.negative_entry.setContextMenuPolicy(Qt.CustomContextMenu)
        self.prompt_entry.customContextMenuRequested.connect(lambda pos: self._show_text_menu(self.prompt_entry, pos))
        self.negative_entry.customContextMenuRequested.connect(lambda pos: self._show_text_menu(self.negative_entry, pos))

    def _show_text_menu(self, widget, pos):
        menu = QMenu()
        menu.addAction("Cut", lambda: widget.cut())
        menu.addAction("Copy", lambda: widget.copy())
        menu.addAction("Paste", lambda: widget.paste())
        menu.exec(widget.mapToGlobal(pos))
        # Model and LoRA population
        self.available_models = self._find_local_models()
        self.model_combo.addItems(list(self.available_models.keys()))
        self.lora_options = self._find_loras()
        self.lora_combo.addItems(self.lora_options)

        # Log app startup
        self.log("App started. Ready to generate an image.")

    def log(self, message):
        self.log_text.append(message)
        self.status_label.setText(message)

    def _find_local_models(self):
        models = {}
        if not os.path.exists(MODELS_DIR):
            return models
        for entry in os.listdir(MODELS_DIR):
            entry_path = os.path.join(MODELS_DIR, entry)
            if os.path.isdir(entry_path):
                has_model_file = False
                for fname in os.listdir(entry_path):
                    if fname.endswith(".ckpt") or fname.endswith(".safetensors") or fname == "model_index.json":
                        has_model_file = True
                        break
                if has_model_file:
                    models[entry] = entry_path
        return models

    def _find_loras(self):
        loras = [""]
        loras_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loras")
        if os.path.isdir(loras_dir):
            for entry in os.listdir(loras_dir):
                entry_path = os.path.join(loras_dir, entry)
                if os.path.isdir(entry_path):
                    loras.append(entry)
        return loras


if __name__ == "__main__":
    app = QApplication([])
    window = SDApp()
    window.show()
    app.exec()
