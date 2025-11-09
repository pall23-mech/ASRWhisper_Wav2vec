# ==============================================================
# 🍬 Whisper inference for Icelandic podcasts (full dataset, save text/confidence, upload HF)
# ==============================================================

!pip install -q torch torchaudio transformers datasets huggingface_hub accelerate tqdm soundfile huggingface_hub[hf_transfer]

import torch, math, os, io
from tqdm import tqdm
from datasets import load_dataset, Dataset, Features, Value, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np
import torchaudio
from huggingface_hub import login

# ---------------------------------------------------------------
# 1️⃣ Config
# ---------------------------------------------------------------
REPO_ID   = "palli23/icelandic-podcasts-split"
MODEL_ID  = "palli23/whisper-large-icelandic-multi-aug-v2"
NEW_REPO  = "palli23/icelandic-podcasts-split-whisper-full"  # New repo for full results
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
REPETITION_PENALTY = 1.2  # Mild anti-repeat

# ---------------------------------------------------------------
# 2️⃣ Load dataset (streaming, full)
# ---------------------------------------------------------------
print(f"⬇️  Loading dataset {REPO_ID} (streaming, full) …")
dataset = load_dataset(REPO_ID, streaming=True)
if "train" in dataset:
    dataset = dataset["train"]
else:
    dataset = dataset[list(dataset.keys())[0]]
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

print(f"📁 Dataset loaded (streaming mode).")

# ---------------------------------------------------------------
# 3️⃣ Load Whisper model
# ---------------------------------------------------------------
print(f"📦 Loading Whisper model {MODEL_ID} on {DEVICE} …")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID).to(DEVICE)

# ---------------------------------------------------------------
# 4️⃣ Confidence from logprobs (mean exp(max logp per token, skip BOS/pad)
# ---------------------------------------------------------------
def compute_confidence(logits):
    if logits is None or len(logits) == 0:
        return 0.0
    # Logprobs: max per token position (skip first for BOS)
    logprobs = []
    for l in logits[1:]:  # Skip first token
        logp = torch.log_softmax(l, dim=-1).max(dim=-1)[0]  # Max logp
        logprobs.extend(logp.tolist())
    if not logprobs:
        return 0.0
    mean_logprob = sum(logprobs) / len(logprobs)
    return round(math.exp(mean_logprob), 3)

# ---------------------------------------------------------------
# 5️⃣ Inference on Full Dataset (streaming loop)
# ---------------------------------------------------------------
results = []
print("🚀 Running Whisper inference on full dataset …")

# Iterate over full streaming dataset (8387 examples from your logs)
for idx, example in enumerate(tqdm(dataset, desc="🔊 Transcribing", total=8387)):
    try:
        audio_info = example["audio"]
        # Manual decode with torchaudio
        if isinstance(audio_info, dict):
            if 'bytes' in audio_info:
                waveform, sr = torchaudio.load(io.BytesIO(audio_info['bytes']))
            elif 'path' in audio_info:
                waveform, sr = torchaudio.load(audio_info['path'])
            else:
                raise ValueError(f"Unsupported audio format: {audio_info.keys()}")
            audio_array = waveform.mean(0).numpy()  # Mono
        else:
            audio_array = np.array(audio_info)

        if len(audio_array) == 0:
            raise ValueError("Empty audio")

        # Preprocess input
        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt").to(DEVICE)
        input_features = inputs.input_features

        # Generate with logprobs + repeat penalty
        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=1,  # Greedy for efficiency
                language="is",
                task="transcribe",
                output_scores=True,  # Token logprobs
                return_dict_in_generate=True,
                do_sample=False,
                repetition_penalty=REPETITION_PENALTY  # Anti-repeat
            )

        # Decode text
        text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0].strip()

        # Extract logprobs
        logits = generated_ids.scores  # List of logits per token
        confidence = compute_confidence(logits)

        results.append({
            "example_id": example.get("id", idx),
            "text": text,
            "confidence": confidence
        })

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"⚠️  Skipped example {example.get('id', idx)}: {e}")
        results.append({
            "example_id": example.get("id", idx),
            "text": "",
            "confidence": 0.0
        })

print(f"✅ Finished all {len(results)} examples.")

# ---------------------------------------------------------------
# 6️⃣ Create Dataset & Save
# ---------------------------------------------------------------
print("🧩 Creating new dataset …")
if len(results) > 0:
    # Define schema
    features = Features({
        "example_id": Value("int32"),
        "text": Value("string"),
        "confidence": Value("float32")
    })
    ds_new = Dataset.from_list(results, features=features)
else:
    ds_new = Dataset.from_list([])

ds_new.save_to_disk("/content/icelandic_podcasts_split_whisper_full")

# ---------------------------------------------------------------
# 7️⃪ Upload to HF
# ---------------------------------------------------------------
print("🔐 Logging in to Hugging Face …")
login()  # Paste your HF token

print(f"☁️  Uploading dataset to {NEW_REPO} …")
ds_new.push_to_hub(NEW_REPO, private=True)
print(f"✅ Upload complete → https://huggingface.co/datasets/{NEW_REPO}")

# ---------------------------------------------------------------
# 8️⃣ Quick Preview
# ---------------------------------------------------------------
non_empty = sum(1 for r in results if r["text"].strip())
avg_conf = np.mean([r["confidence"] for r in results if r["confidence"] > 0]) if any(r["confidence"] > 0 for r in results) else 0.0
print(f"📊 Quick Stats: {non_empty}/{len(results)} non-empty transcripts | Avg Confidence: {avg_conf:.3f}")
if results:
    print("Sample:", results[0])
