# PyStudio Diffusion

A simple Tkinter-based GUI for generating images using local Stable Diffusion and SDXL models. Supports model selection, prompt/negative prompt input, image size, guidance scale, and seed control. All heavy operations (model loading, image generation, display) are threaded for a responsive experience.

## Features
- Local-only model support (no cloud API required)
- Select from multiple models in the `models/` folder
- Adjustable image size, guidance scale, and steps
- Prompt and negative prompt input
- Seed control for reproducibility
- Save generated images
- Responsive UI (threaded operations)

## Requirements
- Python 3.8+
- torch (CUDA recommended for best performance)
- diffusers
- Pillow

Install dependencies:
```sh
pip install torch diffusers pillow
```

## Usage
1. Place your Stable Diffusion or SDXL models in the `models/` folder. Each model should have its own subfolder with required files (`model_index.json`, `unet`, `vae`, `text_encoder`, `tokenizer`). Or, run the `getmodels.py` file to download a few models from Hugging Face. (You will need a token key from Hugging Face.)
2. Run the app:
```sh
python main.py
```
3. Select a model, enter your prompt, negative prompt (if desired), adjust settings, and click "Generate Image".

## License

MIT License

Copyright (c) 2025 QuantumPixelator

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
