import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import os
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import threading

# NOTE: Do not commit model files to git. Add 'models/' to your .gitignore.

MODELS_DIR = "./models"

class SDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyStudio Diffusion")
        self.root.geometry("700x750")
        self.root.resizable(False, False)
        style = ttk.Style()
        style.theme_use("clam")

        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Model and image size section
        top_frame = ttk.LabelFrame(main_frame, text="Model & Image Settings", padding=10)
        top_frame.pack(fill="x", pady=10)

        ttk.Label(top_frame, text="Select Model:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_var = tk.StringVar(root)
        self.model_var.set("")
        self.model_menu = ttk.OptionMenu(top_frame, self.model_var, "")
        self.model_menu.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(top_frame, text="Image Size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.size_var = tk.StringVar(root)
        self.size_options = ["512x512", "768x768", "1024x1024", "1920x1080"]
        self.size_var.set(self.size_options[0])
        self.size_menu = ttk.OptionMenu(top_frame, self.size_var, self.size_options[0], *self.size_options)
        self.size_menu.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # Options section
        options_frame = ttk.LabelFrame(main_frame, text="Generation Options", padding=10)
        options_frame.pack(fill="x", pady=10)

        ttk.Label(options_frame, text="Prompt Adherence / Creativity:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.guidance_var = tk.DoubleVar(root)
        self.guidance_var.set(7.0)
        guidance_frame = ttk.Frame(options_frame)
        guidance_frame.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.guidance_scale = ttk.Scale(guidance_frame, from_=1.0, to=20.0, orient="horizontal", variable=self.guidance_var)
        self.guidance_scale.pack(side="left", fill="x", expand=True)
        self.guidance_value_label = ttk.Label(guidance_frame, text=f"{self.guidance_var.get():.2f}")
        self.guidance_value_label.pack(side="left", padx=5)
        def update_guidance_label(*args):
            self.guidance_value_label.config(text=f"{self.guidance_var.get():.2f}")
        self.guidance_var.trace_add("write", update_guidance_label)
        ttk.Label(options_frame, text="Lower = more creative, Higher = more literal (default: 7.0)", font=("Arial", 8)).grid(row=0, column=2, sticky="w", padx=5)

        ttk.Label(options_frame, text="Quality / Detail (Steps):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.steps_var = tk.IntVar(root)
        self.steps_var.set(30)
        steps_frame = ttk.Frame(options_frame)
        steps_frame.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.steps_scale = ttk.Scale(steps_frame, from_=10, to=100, orient="horizontal", variable=self.steps_var)
        self.steps_scale.pack(side="left", fill="x", expand=True)
        self.steps_value_label = ttk.Label(steps_frame, text=f"{self.steps_var.get()}")
        self.steps_value_label.pack(side="left", padx=5)
        def update_steps_label(*args):
            self.steps_value_label.config(text=f"{self.steps_var.get()}")
        self.steps_var.trace_add("write", update_steps_label)
        ttk.Label(options_frame, text="Higher = more detail, slower (default: 30)", font=("Arial", 8)).grid(row=1, column=2, sticky="w", padx=5)

        ttk.Label(options_frame, text="Random Seed:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        seed_frame = ttk.Frame(options_frame)
        seed_frame.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.seed_entry = ttk.Entry(seed_frame)
        self.seed_entry.pack(side="left", fill="x", expand=True)
        clear_seed_btn = ttk.Button(seed_frame, text="✕", width=2, command=lambda: self.seed_entry.delete(0, tk.END))
        clear_seed_btn.pack(side="left", padx=2)
        ttk.Label(options_frame, text="Set for reproducible results, blank for random", font=("Arial", 8)).grid(row=2, column=2, sticky="w", padx=5)

        options_frame.columnconfigure(1, weight=1)

        # Prompts frame
        prompts_frame = ttk.Frame(main_frame)
        prompts_frame.pack(fill="both", pady=10)

        # Prompt frame
        prompt_frame = ttk.LabelFrame(prompts_frame, text="Image Description (Prompt)", padding=10)
        prompt_frame.pack(side="left", fill="both", expand=True)
        self.prompt_entry = tk.Text(prompt_frame, width=30, height=4, wrap="word")
        self.prompt_entry.pack(side="left", fill="both", expand=True)
        self.prompt_scroll = ttk.Scrollbar(prompt_frame, command=self.prompt_entry.yview)
        self.prompt_scroll.pack(side="right", fill="y")
        self.prompt_entry.config(yscrollcommand=self.prompt_scroll.set)

        # Negative prompt frame
        negative_frame = ttk.LabelFrame(prompts_frame, text="Negative Prompt (optional)", padding=10)
        negative_frame.pack(side="right", fill="both", expand=True)
        self.negative_entry = tk.Text(negative_frame, width=30, height=2, wrap="word")
        self.negative_entry.pack(side="left", fill="both", expand=True)
        self.negative_scroll = ttk.Scrollbar(negative_frame, command=self.negative_entry.yview)
        self.negative_scroll.pack(side="right", fill="y")
        self.negative_entry.config(yscrollcommand=self.negative_scroll.set)

        # Add right-click context menu for copy/paste/cut
        self.text_menu = tk.Menu(root, tearoff=0)
        self.text_menu.add_command(label="Cut", command=lambda: self._text_event("cut"))
        self.text_menu.add_command(label="Copy", command=lambda: self._text_event("copy"))
        self.text_menu.add_command(label="Paste", command=lambda: self._text_event("paste"))
        self.prompt_entry.bind("<Button-3>", lambda e: self._show_text_menu(e, self.prompt_entry))
        self.negative_entry.bind("<Button-3>", lambda e: self._show_text_menu(e, self.negative_entry))

        # Auto-load negative prompt from negative.txt if it exists
        neg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "negative.txt")
        if os.path.isfile(neg_path):
            try:
                with open(neg_path, "r", encoding="utf-8") as nf:
                    neg_text = nf.read().strip()
                self.negative_entry.delete("1.0", "end")
                self.negative_entry.insert("1.0", neg_text)
            except Exception:
                pass
        # Generate button and status
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill="x", pady=10)

        # Generate/Cancel button
        self.generate_button = ttk.Button(bottom_frame, text="Generate Image", command=self.generate_image)
        self.generate_button.pack(side="left", padx=10)
        self.cancel_button = ttk.Button(bottom_frame, text="Cancel", command=self.cancel_generation, state="disabled")
        self.cancel_button.pack(side="left", padx=5)

        # Auto Clear Seed checkbox
        self.auto_clear_seed_var = tk.BooleanVar(value=False)
        self.auto_clear_seed_check = ttk.Checkbutton(bottom_frame, text="Auto Clear Seed", variable=self.auto_clear_seed_var)
        self.auto_clear_seed_check.pack(side="left", padx=10)



        # Progress bar in its own row below the buttons
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill="x", pady=(0, 10))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=400)
        self.progress_bar.pack(fill="x", padx=10)

        self.status_label = ttk.Label(bottom_frame, text="Ready to generate an image", anchor="w", justify="left")
        self.status_label.pack(side="left", padx=10, fill="x", expand=True)

        # Log box
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=10)
        log_frame.pack(fill="both", pady=10, expand=True)
        self.log_text = tk.Text(log_frame, height=50, wrap="word", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_scroll.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=self.log_scroll.set)



        self.pipe = None
        self.current_model = None
        self.is_generating = False
        self._cancel_requested = False

        # Log app startup
        self.log("App started. Ready to generate an image.")

        # Scan models folder for available models
        self.available_models = self._find_local_models()
        if not self.available_models:
            messagebox.showerror("No Models Found", f"No models found in '{MODELS_DIR}'. Please add a model folder.")
            self.model_var.set("")
            self.model_menu['menu'].delete(0, 'end')
            self.generate_button.config(state="disabled")
        else:
            self.model_var.set(list(self.available_models.keys())[0])
            self.model_menu['menu'].delete(0, 'end')
            for name in self.available_models.keys():
                self.model_menu['menu'].add_command(label=name, command=tk._setit(self.model_var, name))
            self.generate_button.config(state="normal")

        # Bind model selection change to log
        def on_model_change(*args):
            model = self.model_var.get()
            if model:
                self.log(f"Model selected: {model}")
            else:
                self.log("No model selected.")
        self.model_var.trace_add("write", on_model_change)

        # Bind image size change to log
        def on_size_change(*args):
            size = self.size_var.get()
            self.log(f"Image size selected: {size}")
        self.size_var.trace_add("write", on_size_change)

    def cancel_generation(self):
        self._cancel_requested = True
        self.log("Image generation cancelled by user.")
        self.status_label.config(text="Generation cancelled.")
        self.generate_button.config(state="normal")
        self.cancel_button.config(state="disabled")

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.status_label.config(text=message)

    def _show_text_menu(self, event, widget):
        widget.focus_set()
        self.text_menu.tk_popup(event.x_root, event.y_root)

    def _text_event(self, action):
        widget = self.root.focus_get()
        try:
            if action == "cut":
                widget.event_generate("<<Cut>>")
            elif action == "copy":
                widget.event_generate("<<Copy>>")
            elif action == "paste":
                widget.event_generate("<<Paste>>")
        except Exception:
            pass

    def _find_local_models(self):
        models = {}
        if not os.path.exists(MODELS_DIR):
            return models
        for entry in os.listdir(MODELS_DIR):
            entry_path = os.path.join(MODELS_DIR, entry)
            if os.path.isdir(entry_path):
                required = [
                    "model_index.json",
                    "unet",
                    "vae",
                    "text_encoder",
                    "tokenizer"
                ]
                missing = [item for item in required if not os.path.exists(os.path.join(entry_path, item))]
                if not missing:
                    models[entry] = entry_path
        return models

    def load_model_threaded(self, callback=None):
        def load():
            model_name = self.model_var.get()
            model_dir = self.available_models.get(model_name, None)
            if not model_name or not model_dir:
                self.root.after(0, lambda: self.log("No model selected or model not found."))
                if callback:
                    self.root.after(0, lambda: callback(None))
                return
            if self.current_model == model_name and self.pipe is not None:
                if callback:
                    self.root.after(0, lambda: callback(self.pipe))
                return
            self.root.after(0, lambda: self.log(f"Loading model: {model_name}"))
            self.root.after(0, self.root.update)
            try:
                use_cuda = torch.cuda.is_available()
                if not use_cuda:
                    self.root.after(0, lambda: self.log("CUDA GPU not detected. Image generation will be much slower and may produce poor results, especially for SDXL."))
                    self.root.after(0, lambda: messagebox.showwarning("CUDA Not Available", "CUDA GPU not detected. Image generation will be much slower and may produce poor results, especially for SDXL."))
                torch_dtype = torch.float16 if use_cuda else torch.float32
                if "xl" in model_name.lower():
                    pipe = StableDiffusionXLPipeline.from_pretrained(model_dir, torch_dtype=torch_dtype, safety_checker=None)
                else:
                    pipe = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch_dtype, safety_checker=None)
                device = "cuda" if use_cuda else "cpu"
                pipe = pipe.to(device)
                self.pipe = pipe
                self.current_model = model_name
                self.root.after(0, lambda: self.log(f"Loaded model: {model_name} (Device: {device})"))
                if callback:
                    self.root.after(0, lambda: callback(pipe))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"Failed to load model: {e}"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {e}"))
                self.root.after(0, lambda: self.status_label.config(text="Model loading failed"))
                if callback:
                    self.root.after(0, lambda: callback(None))
        threading.Thread(target=load, daemon=True).start()

    def generate_image(self):
        if self.is_generating:
            self.log("Image generation in progress! Please wait.")
            messagebox.showwarning("Warning", "Image generation in progress! Please wait.")
            return
        prompt = self.prompt_entry.get("1.0", "end").strip()
        negative_prompt = self.negative_entry.get("1.0", "end").strip()
        model_name = self.model_var.get()
        if not model_name or model_name not in self.available_models:
            self.log("Please select a model before generating an image.")
            messagebox.showwarning("Model Error", "Please select a model before generating an image.")
            return
        if not prompt:
            self.log("Please enter a description!")
            messagebox.showwarning("Input Error", "Please enter a description!")
            return
        self.generate_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.is_generating = True
        self._cancel_requested = False
        # Auto clear seed if box is checked
        if self.auto_clear_seed_var.get():
            self.seed_entry.delete(0, tk.END)
        self.log("Loading model...")
        self.progress_var.set(0)
        self.progress_bar.update()
        def after_model_loaded(pipe):
            if pipe is None:
                self.is_generating = False
                self.root.after(0, lambda: self.generate_button.config(state="normal"))
                self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
                return
            self.log("Generating image...")
            def run_generation():
                def progress_callback(step, timestep, total_steps):
                    percent = int((step + 1) / total_steps * 100)
                    self.root.after(0, lambda: self.progress_var.set(percent))
                    self.root.after(0, self.progress_bar.update)
                try:
                    size_str = self.size_var.get()
                    if "x" in size_str:
                        width, height = map(int, size_str.split("x"))
                    else:
                        width, height = 512, 512
                    guidance_scale = self.guidance_var.get()
                    num_inference_steps = self.steps_var.get()
                    seed_str = self.seed_entry.get().strip()
                    generator = None
                    used_seed = None
                    seed_str = self.seed_entry.get().strip()
                    if seed_str:
                        try:
                            seed = int(seed_str)
                            generator = torch.manual_seed(seed)
                            used_seed = seed
                        except ValueError:
                            self.root.after(0, lambda: messagebox.showwarning("Seed Error", "Seed must be an integer or blank."))
                            generator = None
                    else:
                        import random
                        used_seed = random.randint(0, 2**32 - 1)
                        generator = torch.manual_seed(used_seed)
                    use_cuda = torch.cuda.is_available()
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
                        # Some older diffusers may not support callback_on_step_end
                        try:
                            result = pipe(**pipe_args)
                        except TypeError:
                            # Remove callback_on_step_end if not supported
                            pipe_args.pop('callback_on_step_end', None)
                            try:
                                result = pipe(**pipe_args)
                            except Exception:
                                self.root.after(0, lambda: self.log("Progress bar not supported in this diffusers/model version."))
                                # Fallback: no progress bar
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
                    # Check for cancel
                    if self._cancel_requested:
                        self.root.after(0, lambda: self.log("Generation cancelled before completion."))
                        self.root.after(0, lambda: self.status_label.config(text="Generation cancelled."))
                        self.is_generating = False
                        self.root.after(0, lambda: self.generate_button.config(state="normal"))
                        self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
                        self.root.after(0, lambda: self.progress_var.set(0))
                        return
                    # Check for black image (all pixels zero)
                    if hasattr(image, 'getextrema'):
                        extrema = image.getextrema()
                        if isinstance(extrema, tuple) and all(e == (0, 0) for e in extrema):
                            self.root.after(0, lambda: self.log("Generated image is completely black. This usually means a model or device issue. Try restarting, updating diffusers, or verifying your model files."))
                            self.root.after(0, lambda: messagebox.showwarning(
                                "Black Image Warning",
                                "Generated image is completely black. This usually means a model or device issue. Try restarting, updating diffusers, or verifying your model files."
                            ))
                    def show_image_window_threaded():
                        def show_image_window():
                            win = tk.Toplevel(self.root)
                            win.title("Generated Image")
                            img_disp = image.copy()
                            img_disp.thumbnail((800, 800), Image.Resampling.LANCZOS)
                            img_tk2 = ImageTk.PhotoImage(img_disp)
                            lbl = tk.Label(win, image=img_tk2)
                            lbl.image = img_tk2
                            lbl.pack()
                            def save_image():
                                filename = filedialog.asksaveasfilename(
                                    defaultextension=".png",
                                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                                    title="Save Generated Image"
                                )
                                if filename:
                                    image.save(filename)
                                    self.log(f"Image saved as {os.path.basename(filename)}")
                                else:
                                    self.log("Image save cancelled")
                            img_menu = tk.Menu(win, tearoff=0)
                            img_menu.add_command(label="Save Image", command=save_image)
                            def show_img_menu(event):
                                img_menu.tk_popup(event.x_root, event.y_root)
                            lbl.bind("<Button-3>", show_img_menu)
                        self.root.after(0, show_image_window)
                    threading.Thread(target=show_image_window_threaded, daemon=True).start()
                    # Always show the used seed after generation
                    self.root.after(0, lambda: self.seed_entry.delete(0, tk.END))
                    self.root.after(0, lambda: self.seed_entry.insert(0, str(used_seed)))
                    self.root.after(0, lambda: self.log("Image generated. Right-click the image to save."))
                    self.root.after(0, lambda: self.progress_var.set(100))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to generate image: {e}"))
                    self.root.after(0, lambda: self.log(f"Image generation failed: {e}"))
                    self.root.after(0, lambda: self.progress_var.set(0))
                finally:
                    self.is_generating = False
                    self.root.after(0, lambda: self.generate_button.config(state="normal"))
                    self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
            threading.Thread(target=run_generation, daemon=True).start()
        self.load_model_threaded(callback=after_model_loaded)

if __name__ == "__main__":
    root = tk.Tk()
    app = SDApp(root)
    root.mainloop()