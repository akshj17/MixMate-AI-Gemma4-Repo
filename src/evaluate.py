"""
MixMate AI (CodeMix AI) — 3-Way Evaluation Script (chrF++ & Script Purity)
Benchmarks Plain Gemma 4 E2B (Zero-Shot & 3-Shot) vs. Fine-Tuned CodeMix LoRA
on the locked 45-sentence test split from ai4bharat/IndicCMix (hi.parquet).
"""

import argparse
import gc
import os
import re
import time
import pandas as pd
import sacrebleu

TASK_PROMPTS = {
    "codemixed_devanagari": (
        "Convert the following English sentence into natural code-mixed Hindi "
        "(use Devanagari script for Hindi grammar, but keep technical/common English nouns in Latin script)."
    ),
    "full_native_devanagari": (
        "Convert the following English sentence into colloquial Hindi written entirely in Devanagari script "
        "(transliterate common English loanwords into Devanagari rather than using overly formal Sanskritized words)."
    ),
    "romanized_casual": (
        "Convert the following English sentence into casual WhatsApp-style Romanized Hindi (Hinglish) using only Latin script."
    ),
    "reverse_to_english": (
        "Convert the following casual Romanized Hinglish message into clean, professional English."
    ),
}

FEW_SHOT_PREFIX = {
    "codemixed_devanagari": (
        "Examples:\n"
        "English: Please share the presentation link before the meeting starts.\n"
        "Output: मीटिंग शुरू होने से पहले कृपया presentation link शेयर कर दें।\n"
        "English: Our developer team pushed the bug fix to the production server.\n"
        "Output: हमारी developer team ने production server पर bug fix पुश कर दिया है।\n\n"
    ),
    "full_native_devanagari": (
        "Examples:\n"
        "English: Please share the presentation link before the meeting starts.\n"
        "Output: मीटिंग शुरू होने से पहले कृपया प्रेजेंटेशन लिंक शेयर कर दें।\n"
        "English: Our developer team pushed the bug fix to the production server.\n"
        "Output: हमारी डेवलपर टीम ने प्रोडक्शन सर्वर पर बग फिक्स पुश कर दिया है।\n\n"
    ),
    "romanized_casual": (
        "Examples:\n"
        "English: Please share the presentation link before the meeting starts.\n"
        "Output: Meeting shuru hone se pehle please presentation link share kar dena.\n"
        "English: Our developer team pushed the bug fix to the production server.\n"
        "Output: Hamari developer team ne production server par bug fix push kar diya hai.\n\n"
    ),
}


def check_script_compliance(text: str, task: str) -> float:
    """Validates whether the generated output obeys the target script rules."""
    text = str(text or "")
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if task == "codemixed_devanagari":
        return 100.0 if (has_devanagari and has_latin) else 0.0
    elif task == "full_native_devanagari":
        return 100.0 if (has_devanagari and not has_latin) else 0.0
    elif task == "romanized_casual":
        return 100.0 if (has_latin and not has_devanagari) else 0.0
    return 100.0


def prepare_test_split(output_csv: str, hf_token: str | None = None) -> pd.DataFrame:
    """Reconstructs the exact 45-sentence locked test split (indices 3050:3095, seed=42) from ai4bharat/IndicCMix."""
    from datasets import load_dataset

    raw_ds = load_dataset(
        "ai4bharat/IndicCMix",
        data_files="hi.parquet",
        split="train",
        token=hf_token,
    )
    if "audio" in raw_ds.column_names:
        raw_ds = raw_ds.remove_columns(["audio"])

    def is_clean(row):
        if (
            not row["english"]
            or not row["native_script_codemixed"]
            or not row["full_native_script"]
            or not row["romanized_casual"]
        ):
            return False
        wc = len(row["english"].split())
        return 8 <= wc <= 35

    clean_ds = raw_ds.filter(is_clean, num_proc=2).shuffle(seed=42)
    test_raw = clean_ds.select(range(3050, 3095))

    modes = [
        ("codemixed_devanagari", "native_script_codemixed"),
        ("full_native_devanagari", "full_native_script"),
        ("romanized_casual", "romanized_casual"),
    ]
    test_rows = []
    for idx, row in enumerate(test_raw):
        mode_key, col_name = modes[idx % 3]
        test_rows.append(
            {
                "id": row["id"],
                "task": mode_key,
                "english": row["english"].strip(),
                "reference": row[col_name].strip(),
            }
        )

    test_df = pd.DataFrame(test_rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    test_df.to_csv(output_csv, index=False)
    print(f"Saved locked 45-sentence test split to {output_csv}")
    return test_df


def generate_single(model, tokenizer, user_prompt: str, use_adapter: bool = True):
    """Generates a single response with or without the LoRA adapter enabled."""
    import torch

    messages = [{"role": "user", "content": user_prompt}]
    try:
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        messages_struct = [
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages_struct, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(text=prompt_str, return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    with torch.no_grad():
        if use_adapter or not hasattr(model, "disable_adapter"):
            out_ids = model.generate(
                **inputs, max_new_tokens=96, temperature=0.2, do_sample=False
            )
        else:
            with model.disable_adapter():
                out_ids = model.generate(
                    **inputs, max_new_tokens=96, temperature=0.2, do_sample=False
                )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    gen_tokens = out_ids[0][inputs["input_ids"].shape[1] :]
    pred = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    pred = re.sub(r"<\|channel\>.*?<channel\|>", "", pred, flags=re.DOTALL).strip()
    lines = [ln.strip() for ln in pred.split("\n") if ln.strip()]
    pred = lines[0] if lines else pred
    pred = pred.replace("Output:", "").replace("Answer:", "").strip()
    return pred, latency_ms


def compute_scorecard(res_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates corpus-level chrF++ (word_order=2), script purity, and latency by task mode."""
    summary_rows = []
    for task in ["codemixed_devanagari", "full_native_devanagari", "romanized_casual"]:
        sub = res_df[res_df["task"] == task]
        if sub.empty:
            continue
        refs = [sub["reference"].astype(str).tolist()]

        chrf_zero = sacrebleu.corpus_chrf(
            sub["zero_shot_pred"].astype(str).tolist(), refs, word_order=2
        ).score
        chrf_few = sacrebleu.corpus_chrf(
            sub["few_shot_pred"].astype(str).tolist(), refs, word_order=2
        ).score
        chrf_ft = sacrebleu.corpus_chrf(
            sub["finetuned_pred"].astype(str).tolist(), refs, word_order=2
        ).score

        sc_zero = (
            sub["zero_shot_pred"]
            .apply(lambda x: check_script_compliance(x, task))
            .mean()
        )
        sc_ft = (
            sub["finetuned_pred"]
            .apply(lambda x: check_script_compliance(x, task))
            .mean()
        )

        summary_rows.append(
            {
                "Task Mode": task,
                "Base Zero-Shot chrF++": round(chrf_zero, 2),
                "Base 3-Shot chrF++": round(chrf_few, 2),
                "LoRA Fine-Tuned chrF++": round(chrf_ft, 2),
                "Delta vs Zero-Shot": f"+{round(chrf_ft - chrf_zero, 2)}",
                "Base Script Purity (%)": round(sc_zero, 1),
                "LoRA Script Purity (%)": round(sc_ft, 1),
                "LoRA Avg Latency (ms)": round(sub["finetuned_lat_ms"].mean(), 0),
            }
        )
    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run 3-way chrF++ and Script Purity evaluation for MixMate AI."
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=os.environ.get("ADAPTER_PATH", "codemix_gemma4_lora"),
        help="Path or Hugging Face repo ID for the fine-tuned LoRA adapter.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data_samples/test_45_locked.csv",
        help="Path to the locked 45-sentence test CSV.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="results/detailed_45_predictions.csv",
        help="Path to save or read detailed predictions across all 45 test rows.",
    )
    parser.add_argument(
        "--scorecard-csv",
        type=str,
        default="results/evaluation_scorecard.csv",
        help="Path to save the quantitative benchmark scorecard CSV.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Recompute scorecard metrics from --predictions-csv without running GPU inference.",
    )
    args = parser.parse_args()

    if args.score_only:
        res_df = pd.read_csv(args.predictions_csv)
        scorecard_df = compute_scorecard(res_df)
        os.makedirs(os.path.dirname(os.path.abspath(args.scorecard_csv)), exist_ok=True)
        scorecard_df.to_csv(args.scorecard_csv, index=False)
        print(scorecard_df.to_string(index=False))
        return

    if os.path.exists(args.test_csv) and len(pd.read_csv(args.test_csv)) == 45:
        test_df = pd.read_csv(args.test_csv)
    else:
        hf_token = os.environ.get("HF_TOKEN")
        test_df = prepare_test_split(args.test_csv, hf_token=hf_token)

    import torch
    from unsloth import FastModel

    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.adapter_path,
        dtype=None,
        max_seq_length=1024,
        load_in_4bit=True,
        full_finetuning=False,
        text_only=True,
    )

    eval_records = []
    for _, row in test_df.iterrows():
        task = row["task"]
        eng = row["english"]
        ref = row["reference"]

        base_prompt = (
            f"{TASK_PROMPTS[task]}\nOutput ONLY the converted sentence.\n\nEnglish: {eng}"
        )
        fewshot_prompt = (
            f"{TASK_PROMPTS[task]}\n{FEW_SHOT_PREFIX[task]}"
            f"Output ONLY the converted sentence.\nEnglish: {eng}"
        )
        ft_prompt = f"{TASK_PROMPTS[task]}\n\nEnglish: {eng}"

        pred_zero, lat_zero = generate_single(
            model, tokenizer, base_prompt, use_adapter=False
        )
        pred_few, lat_few = generate_single(
            model, tokenizer, fewshot_prompt, use_adapter=False
        )
        pred_ft, lat_ft = generate_single(
            model, tokenizer, ft_prompt, use_adapter=True
        )

        eval_records.append(
            {
                "id": row["id"],
                "task": task,
                "english": eng,
                "reference": ref,
                "zero_shot_pred": pred_zero,
                "zero_shot_lat_ms": lat_zero,
                "few_shot_pred": pred_few,
                "few_shot_lat_ms": lat_few,
                "finetuned_pred": pred_ft,
                "finetuned_lat_ms": lat_ft,
            }
        )

    res_df = pd.DataFrame(eval_records)
    os.makedirs(os.path.dirname(os.path.abspath(args.predictions_csv)), exist_ok=True)
    res_df.to_csv(args.predictions_csv, index=False)

    scorecard_df = compute_scorecard(res_df)
    os.makedirs(os.path.dirname(os.path.abspath(args.scorecard_csv)), exist_ok=True)
    scorecard_df.to_csv(args.scorecard_csv, index=False)
    print(scorecard_df.to_string(index=False))


if __name__ == "__main__":
    main()
