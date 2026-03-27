# 🚀 Nemotron-30B: The Blackwell MoE Optimizer

An advanced fine-tuning pipeline for the **Nemotron-3-Nano-30B** (Mixture of Experts) model, specifically optimized for **NVIDIA Blackwell** architecture in restricted, offline environments.

---

## 📖 Overview

Fine-tuning a 30B Mixture of Experts (MoE) model on a single GPU is a significant engineering challenge. This project provides a robust solution for training the Nemotron-30B architecture within an **offline Kaggle environment**, addressing critical hardware-level precision conflicts and memory constraints.

## 🛠️ The Technical Challenge

During the development of this pipeline, we encountered and solved three "Final Boss" bugs inherent to the Nemotron/Blackwell/Unsloth stack:

1.  **Dtype Promotion Mismatch:** Nemotron’s custom MoE router calculates weights in `Float32`, while the Experts run in `BFloat16`. This leads to a `RuntimeError` during the `index_add_` operation.
2.  **The 8-Bit Dummy Pass:** In quantized modes (4-bit/8-bit), the model triggers a "dummy pass" for empty experts using `unsigned char` (uint8) weights, which conflicts with `BFloat16` LoRA inputs.
3.  **DNA Persistence Loops:** Standard Python patching causes `RecursionError` due to class-level function caching in the Kaggle notebook environment.

## ✨ The Solution: "The Golden Bridge"

We implemented a **DNA-level Class Patch** that surgically rewrites the model's core MoE logic in RAM. This "Golden Bridge" intercepts routing weights and dummy inputs, forcing them into `BFloat16` decimals before they hit the calculation kernel.

### Key Engineering Features:
* **Idempotent DNA Patching:** Recursion-proof logic that safely modifies class blueprints without infinite loops.
* **Precision Purge:** Forcible alignment of LoRA parameters to `BFloat16` to bypass PEFT upcasting bugs.
* **Blackwell Memory Tuning:** Optimized gradient accumulation (8 steps) and per-device batch size (1) to maximize the 94.9 GB VRAM of the RTX PRO 6000.
* **100% Offline Compatibility:** Bootstrapped via local `.whl` files for Unsloth and Triton.

---

## 🏗️ Technical Stack

| Component | Technology |
| :--- | :--- |
| **Model** | Nemotron-3-Nano-30B (MoE) |
| **GPU** | NVIDIA RTX PRO 6000 (Blackwell Architecture) |
| **Framework** | PyTorch + Hugging Face Transformers |
| **Optimization** | Unsloth + PEFT (LoRA) |
| **Quantization** | BitsAndBytes (8-bit / 4-bit) |

---

## 🚀 Execution Pipeline

1.  **Environment Initialization:** Load local Unsloth/Triton wheels.
2.  **Model Loading:** Initialize the 30B Beast with `load_in_4bit=True`.
3.  **DNA Patching:** Execute the "Ironclad" patches to fix the MoE Router and Expert blueprints.
4.  **Tokenization:** Map the dataset using the Nemotron-specific tokenizer.
5.  **Ignition:** Start the `transformers.Trainer` loop.

> [!IMPORTANT]
> **A Note on Recursion:** If the notebook crashes, a **Factory Reset** is mandatory to clear the "poisoned" class definitions from the Python interpreter memory.

---

## 📝 Author

**Aman Singh** *UX Designer, AI Researcher, and Builder* Founder of **UXterity** 📍 Delhi, India

---

## ⚖️ License
This project is developed for the [Competition Name] and follows the licensing terms of the Nemotron-3-Nano-30B model.
