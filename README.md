# Nemotron Reasoning Challenge Optimization System

This repository dynamically bootstraps, filters, trains, and verifies a complete self-refinement and instruction-tuning stack targeting reasoning tasks aligned with the NVIDIA Nemotron Challenge.

## 🏗️ Architecture Stack

1. **Data Engine (`src/data_engine/`)**
   - Implements automated logical trace bootstrapping.
   - Assesses syntax completion and ground-truth mapping to enforce high-confidence samples.

2. **Prompt Engineering Engine (`src/prompt_engine/`)**
   - Employs rigorously optimized Chain-of-Thought (CoT) paths requiring final answers encased in `\boxed{}`.
   - Facilitates temperature-sampled **Self-Consistency (Majority Vote)** sampling trajectories.

3. **Self-Refinement System (`src/self_refinement/`)**
   - Establishes a multi-turn adversarial environment for the model to critique and dynamically rewrite its own flawed pathways before adding them to the training dataset.

4. **Training Engine (`src/training_engine/`)**
   - Efficient QLoRA wrapper explicitly configured for memory-optimized (BitsAndBytes 4-Bit) environments training up to Rank 32 constraint compliance using `SFTTrainer`.

5. **Evaluation Engine (`src/evaluation_engine/`)**
   - Customized Regex extraction engine ensuring competition-grade exact match equivalency on math outputs alongside symbolic normalizers.

6. **Optimization & Advanced Techniques (`src/optimization/`)**
   - Contains:
     - **Curriculum Learning**: Progressing sample difficulty logically through training streams.
     - **Error Clustering**: Pattern tracking for logical failures (LaTeX vs Syntax vs Hallucination).
     - **Trace Distillation**: Overriding majority variance by choosing the tightest execution paths computationally. 
     - **Dynamic Data Scoring (Confidence)**: Ranking dataset pairs through logical structural heuristics algorithmically.

## 🚀 Usage

Execute the complete end-to-end iteration pipeline which establishes baselines, builds mock trace data, constructs LORAs, and verifies iteratively:
```bash
python src/pipeline.py --model nvidia/Nemotron-3-Nano-30B --epochs 3
```

To merge a successfully trained LoRA checkpoint backward onto the base model payload for immediate unquantized **vLLM deployment** (submission grade):
```bash
python scripts/export_lora.py --base nvidia/Nemotron-3-Nano-30B --lora lora_adapter_epoch_2 --out submission_model_vLLM
```
