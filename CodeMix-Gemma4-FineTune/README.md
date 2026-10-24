# MixMate AI (CodeMix AI): Fine-Tuned Gemma 4 E2B Multimodal & Cited RAG Suite

**Hacktoberfest Hack Day — Theme 3A + Multimodal & Cited RAG Extensions** | Parameter-Efficient QLoRA Fine-Tuning of `unsloth/gemma-4-E2B-it` on `ai4bharat/IndicCMix`

[![Open Updated Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TviD2h0K4LCUzQyUr3r0xhHN8-Raz_j-?usp=sharing)
[![Open Baseline Training Run In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1F5RJHF3FoHJRSB1d34uMOvKde-O2O4XC?usp=sharing)
[![Dataset](https://img.shields.io/badge/Dataset-ai4bharat%2FIndicCMix-blue)](https://huggingface.co/datasets/ai4bharat/IndicCMix)
[![Base Model](https://img.shields.io/badge/Base%20Model-unsloth%2Fgemma--4--E2B--it-green)](https://huggingface.co/unsloth/gemma-4-E2B-it)

## Overview

Standard instruction-tuned Large Language Models struggle with natural Indian code-switching (Hinglish). When converting English sentences into conversational Hindi, off-the-shelf models either produce overly formal, Sanskritized Hindi or violate script purity rules—yielding only **13.3% script compliance** on mixed Devanagari + Latin technical text and **60.0% script compliance** on colloquial pure Devanagari in zero-shot testing.

**MixMate AI (CodeMix AI)** solves this through a multi-task Supervised Fine-Tuning (SFT) pipeline using **Unsloth 4-bit QLoRA** on a single free Google Colab Tesla T4 GPU, and extends the fine-tuned adapter into a **5-Tab Multimodal & Cited RAG Help Desk Suite**:

1. **Core 4-Mode Code-Mix Fine-Tuning:** Trains **0.60% (31.04M)** of `unsloth/gemma-4-E2B-it` parameters for 260 steps (~15.5 minutes) on 9,715 multi-task examples from `ai4bharat/IndicCMix`, achieving **100.0% script compliance** across all modes and up to a **+27.01 chrF++** gain over zero-shot Gemma 4 E2B.
2. **Cited Bilingual RAG Help Bot:** Combines a CPU-resident `SentenceTransformer("all-MiniLM-L6-v2")` vector index (with Hinglish query expansion and `.pdf`/`.md`/`.txt` ingestion) with single-VRAM adapter toggling—using base Gemma 4 (`use_adapter=False`) for grounded English synthesis with `[Cited Source]` attribution and the fine-tuned LoRA adapter (`use_adapter=True`) for natural Hindi/Hinglish delivery, complete with safe out-of-scope refusal (`similarity < 0.25`).
3. **Multimodal Voice ASR & Handwritten Photo OCR Pipelines:** Integrates Gemma 4's native Audio and Vision encoders (`use_adapter=False`) to transcribe spoken audio clips (16 kHz via `librosa`) and handwritten notes (`PIL`), piping the extracted text directly into the fine-tuned CodeMix LoRA adapter (`use_adapter=True`).

## Supported Linguistic Conversion Modes

| Task Mode Key | UI Label | Description | Script Compliance Rule |
| :--- | :--- | :--- | :--- |
| `codemixed_devanagari` | **1. Code-Mixed Hindi (Devanagari + English Tech Terms)** | Converts English into natural code-mixed Hindi (Devanagari script for Hindi grammar, Latin script for technical/common English nouns). | Must contain **both** Devanagari (`[\u0900-\u097F]`) and Latin (`[A-Za-z]`) characters. |
| `full_native_devanagari` | **2. Full Native Script (Colloquial Devanagari)** | Converts English into colloquial Hindi written entirely in Devanagari script (transliterating English loanwords rather than using stiff Sanskritized words). | Must contain **only** Devanagari (`[\u0900-\u097F]`) and zero Latin characters. |
| `romanized_casual` | **3. Casual Romanized Hinglish (WhatsApp / Latin Script)** | Converts English into casual WhatsApp-style Romanized Hindi (Hinglish) using only Latin script. | Must contain **only** Latin (`[A-Za-z]`) and zero Devanagari characters. |
| `reverse_to_english` | **4. BONUS: Reverse Hinglish -> Clean Professional English** | *(Trained on 25% of training rows)* Converts casual Romanized Hinglish messages back into clean, professional English. | Fluent English in Latin script. |

## Quantitative Benchmark Results (`N = 45` Held-Out Test Split)

Evaluated on the locked 45-sentence test split (`data_samples/test_45_locked.csv`, 15 sentences per core forward mode) comparing **Base Zero-Shot** (`model.disable_adapter()`), **Base 3-Shot** (`FEW_SHOT_PREFIX`), and **LoRA Fine-Tuned** (`codemix_gemma4_lora`) using corpus-level `chrF++` (`sacrebleu.corpus_chrf` with `word_order=2`), automated regex Script Purity, and per-sentence inference latency on a Tesla T4 GPU:

| Task Mode | Base Zero-Shot chrF++ | Base 3-Shot chrF++ | LoRA Fine-Tuned chrF++ | Delta vs. Zero-Shot | Base Script Purity (%) | LoRA Script Purity (%) | LoRA Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `codemixed_devanagari` | 45.36 | 61.11 | **72.37** | **+27.01** | 13.3% | **100.0%** | 6,470 ms |
| `full_native_devanagari` | 52.33 | 61.15 | **71.18** | **+18.85** | 60.0% | **100.0%** | 6,910 ms |
| `romanized_casual` | 57.56 | 59.57 | **63.31** | **+5.76** | 100.0% | **100.0%** | 5,151 ms |

## Qualitative Examples

- **Mode 1 (`codemixed_devanagari`)**
  - **Input:** `The Indian festival organizers partnered with a fintech company to introduce digital payment options like Paytm and UPI for ticket bookings.`
  - **Base Zero-Shot:** `भारतीय उत्सव आयोजकों ने टिकट बुकिंग के लिए पेटीएम और यूपीआई जैसे डिजिटल भुगतान विकल्प पेश करने के लिए एक फिनटेक कंपनी के साथ साझेदारी की।` *(Violates mixed-script constraint)*
  - **Fine-Tuned LoRA:** `भारतीय festival organizers ने एक fintech company के साथ partnership की है ताकि ticket bookings के लिए Paytm और UPI जैसे digital payment options शुरू कर सकें।`
- **Mode 2 (`full_native_devanagari`)**
  - **Input:** `Our developer team pushed the bug fix to the production server.`
  - **Fine-Tuned LoRA:** `हमारी डेवलपर टीम ने प्रोडक्शन सर्वर पर बग फिक्स पुश कर दिया है।`
- **Mode 3 (`romanized_casual`)**
  - **Input:** `Please remind the developer team to deploy the updated payment gateway before tomorrow morning's client meeting.`
  - **Fine-Tuned LoRA:** `Kal subah ki client meeting se pehle developer team ko updated payment gateway deploy karne ke liye remind kar dena.`
- **Mode 4 (`reverse_to_english`)**
  - **Input:** `Kal subah client meeting se pehle backend server par hotfix deploy kar dena warna production down ho jayega.`
  - **Fine-Tuned LoRA:** `Please deploy the hotfix to the backend server before tomorrow morning's client meeting, otherwise production will go down.`

## 5-Tab Gradio Multimodal & RAG Suite (`src/app.py`)

| Tab | Feature | Architecture & Pipeline |
| :--- | :--- | :--- |
| **1. Text Code-Mix Arena (Core 3A)** | Side-by-side baseline vs. fine-tuned model comparison | Runs a single 4-bit model in VRAM, toggling `model.disable_adapter()` for the baseline vs. active LoRA weights, computing live latency (`ms`), regex Script Compliance (`%`), and sentence `chrF++`. |
| **2. Cited RAG Help Bot** | Source-grounded support bot in Hindi & Hinglish | Uses CPU `SentenceTransformer("all-MiniLM-L6-v2")` over pre-loaded docs (`UPI_Payment_Help_Guide.md`, `OpenSource_Contributor_Guide.md`, `Cloud_Server_Deployment_SOP.txt`) + uploaded `.pdf`/`.md`/`.txt` files. Includes Hinglish query token expansion, safe out-of-scope refusal (`score < 0.25`), base-model grounded synthesis, and LoRA style conversion with `[Cited Source]` tags. |
| **3. Voice / Audio to CodeMix** | Speech-to-CodeMix translation | Resamples microphone/uploaded audio to 16 kHz mono (max 30s) via `librosa`, transcribes via Gemma 4's Audio Tower (`use_adapter=False`), and translates via Fine-Tuned CodeMix LoRA (`use_adapter=True`). |
| **4. Handwritten Photo OCR to CodeMix** | Handwritten note OCR + CodeMix translation | Generates a 1-click sample handwritten note via `PIL.ImageDraw` or accepts uploaded photos, extracts text via Gemma 4's Vision Tower (`use_adapter=False`), and converts via Fine-Tuned CodeMix LoRA (`use_adapter=True`). |
| **5. Benchmark Scorecard (`N=45`)** | Held-out evaluation viewer | Displays `results/evaluation_scorecard.csv` and sample rows from `results/detailed_45_predictions.csv`. |

## Dataset Engineering & QLoRA Training Setup

- **Selective Dataset Ingestion (`ai4bharat/IndicCMix`):** Downloads exclusively `hi.parquet` (104,800 Hindi rows, 41.4 MB) and drops the `audio` column metadata immediately to bypass 31 GB of raw audio files.
- **Filtering & Deterministic Splitting:** Retains rows with non-null values across all four text columns (`english`, `native_script_codemixed`, `full_native_script`, `romanized_casual`) and 8–35 English words (`seed=42`). Locks splits at **3,000 train rows** (expanded into **9,715 multi-task SFT examples**), **50 validation rows** (**150 SFT examples**), and **45 held-out test rows** (15 per core mode).
- **VRAM & Precision Optimizations:** Loads `unsloth/gemma-4-E2B-it` in 4-bit (`load_in_4bit=True`, `text_only=True` for pure text SFT) with explicit garbage collection (`gc.collect()`, `torch.cuda.empty_cache()`), automatic duplicate BOS token removal, `float32` training precision on Tesla T4, and Unsloth padding-free batching.
- **LoRA & SFTTrainer Hyperparameters:** `r = 16`, `lora_alpha = 16`, `lora_dropout = 0`, `bias = "none"`, targeting `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (`31,039,488` trainable parameters / `5,154,217,504` total = `0.60%`). Trained via `TRL SFTTrainer` for `260` steps (`per_device_train_batch_size = 4`, `gradient_accumulation_steps = 4`, effective batch size `16`, `max_seq_length = 256`, `optim = "adamw_8bit"`, `learning_rate = 2e-4`, `warmup_steps = 15`, `lr_scheduler_type = "cosine"`, `weight_decay = 0.01`), converging from loss `4.8928` (step 10) to `1.0716` (step 260) in 15m 28s.

## Repository Structure

```text
├── README.md                              <- Project overview and benchmark results
├── requirements.txt                       <- Reproducible Python dependencies
├── notebooks/
│   └── 03a_CodeMix_Gemma4_FineTune.ipynb  <- Fully executed Colab notebook
├── src/
│   ├── app.py                             <- Gradio side-by-side comparison interface
│   └── evaluate.py                        <- chrF++ and Script Purity evaluation script
├── data_samples/
│   └── test_45_locked.csv                 <- 45-sentence held-out test split
├── results/
│   ├── evaluation_scorecard.csv           <- Quantitative benchmark results
│   └── detailed_45_predictions.csv        <- Full prediction outputs across all 45 test rows
└── adapter_config/
    └── adapter_config.json                <- LoRA hyperparameter configuration
```

## Quickstart & Reproducibility

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the 3-Way Evaluation Script (`src/evaluate.py`)

To run the full GPU evaluation across the 45-sentence test split (regenerating `data_samples/test_45_locked.csv`, `results/detailed_45_predictions.csv`, and `results/evaluation_scorecard.csv`), set your `HF_TOKEN` after accepting the dataset terms at [`ai4bharat/IndicCMix`](https://huggingface.co/datasets/ai4bharat/IndicCMix):

```bash
export HF_TOKEN="your_huggingface_token"
python src/evaluate.py --adapter-path ./codemix_gemma4_lora
```

To recompute the corpus-level `chrF++` and Script Purity scorecard directly from an existing predictions CSV without a GPU:

```bash
python src/evaluate.py --score-only --predictions-csv results/detailed_45_predictions.csv
```

### 3. Launch the 5-Tab Gradio Suite (`src/app.py`)

```bash
python src/app.py
```
