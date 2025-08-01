import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from huggingface_hub import snapshot_download, login

# Load Hugging Face token
with open('token.txt') as f:
    TOKEN = f.read().strip()
login(TOKEN)

# 10 popular SD-compatible LoRA models (curated for realism, creativity, and compatibility)
LORAS = [
    
]

DEST_ROOT = "loras"

# Create loras folder if it doesn't exist
if not os.path.isdir(DEST_ROOT):
    os.makedirs(DEST_ROOT)

# Initialize GUI
root = tk.Tk()
root.title("Stable Diffusion LoRA Downloader (Verified)")
frame = ttk.Frame(root, padding=10)
frame.pack(fill="both", expand=True)

def download_lora(lora_id, btn):
    dest = os.path.join(DEST_ROOT, lora_id.replace('/', '_'))
    btn.config(state="disabled")

    # Indeterminate progress bar
    pbar = ttk.Progressbar(frame, orient="horizontal", length=300, mode="indeterminate")
    pbar.pack(pady=2)
    pbar.start(10)

    def run():
        try:
            snapshot_download(
                repo_id=lora_id,
                local_dir=dest,
                token=TOKEN,
                local_dir_use_symlinks=False
            )
            messagebox.showinfo("Success", f"Downloaded: {lora_id}")
        except Exception as e:
            messagebox.showerror("Error", f"{lora_id}\n{str(e)}")
            btn.config(state="normal")
        finally:
            pbar.stop()
            pbar.destroy()

    threading.Thread(target=run, daemon=True).start()

# Build LoRA rows
for l in LORAS:
    row = ttk.Frame(frame)
    row.pack(fill="x", pady=1)
    
    ttk.Label(row, text=l, width=45, anchor="w").pack(side="left")
    btn = ttk.Button(row, text="Download", width=12)
    btn.pack(side="right")

    dest_path = os.path.join(DEST_ROOT, l.replace('/', '_'))
    if os.path.isdir(dest_path):
        btn.config(text="Downloaded", state="disabled")
    else:
        btn.config(command=lambda l=l, b=btn: download_lora(l, b))

root.mainloop()
