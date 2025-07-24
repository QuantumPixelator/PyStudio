import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from huggingface_hub import snapshot_download, login

# Load Hugging Face token
with open('token.txt') as f:
    TOKEN = f.read().strip()
login(TOKEN)

# Popular community-created full SD checkpoints (replace with actual IDs from HF)
MODELS = [
    "SG161222/Realistic_Vision_V5.1_noVAE",     # Realistic Vision
    "Lykon/ChilloutMix",                        # ChilloutMix
    "Lykon/MajicMixRealistic",                  # MajicMix
    "SG161222/CyberRealistic",                  # CyberRealistic
    "stablediffusionapi/epicrealism",           # EpicRealism
    "Lykon/DreamShaper",                        # DreamShaper
    "digiplay/AbsoluteReality_v1.8.1",          # AbsoluteReality
    "RunDiffusion/Juggernaut-XL-v8",            # Juggernaut XL
    "stablediffusionapi/mobius-xl",             # Mobius XL
    "leosam/helloworldxl",                      # HelloWorld XL
]

DEST_ROOT = "models"

# Initialize GUI
root = tk.Tk()
root.title("Stable Diffusion Realism Model Downloader")
frame = ttk.Frame(root, padding=10)
frame.pack(fill="both", expand=True)

def download_model(model_id, btn):
    dest = os.path.join(DEST_ROOT, model_id.replace('/', '_'))
    btn.config(state="disabled")

    # Indeterminate progress bar
    pbar = ttk.Progressbar(frame, orient="horizontal", length=300, mode="indeterminate")
    pbar.pack(pady=2)
    pbar.start(10)

    def run():
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=dest,
                token=TOKEN,
                local_dir_use_symlinks=False
            )
            messagebox.showinfo("Success", f"Downloaded: {model_id}")
        except Exception as e:
            messagebox.showerror("Error", f"{model_id}\n{str(e)}")
            btn.config(state="normal")
        finally:
            pbar.stop()
            pbar.destroy()

    threading.Thread(target=run, daemon=True).start()

# Build model rows
for m in MODELS:
    row = ttk.Frame(frame)
    row.pack(fill="x", pady=1)
    
    ttk.Label(row, text=m, width=45, anchor="w").pack(side="left")
    btn = ttk.Button(row, text="Download", width=12)
    btn.pack(side="right")

    dest_path = os.path.join(DEST_ROOT, m.replace('/', '_'))
    if os.path.isdir(dest_path):
        btn.config(text="Downloaded", state="disabled")
    else:
        btn.config(command=lambda m=m, b=btn: download_model(m, b))

root.mainloop()
