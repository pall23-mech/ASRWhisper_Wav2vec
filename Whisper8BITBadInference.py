# ==============================================================
# 🍬 Faster Whisper inference for Icelandic podcasts (10 examples test, fixed confidence)
# ==============================================================
#8BIT-WORKING-BADLY
!pip install -q torch torchaudio transformers datasets huggingface_hub accelerate tqdm soundfile jiwer huggingface_hub[hf_transfer] faster-whisper ctranslate2

import torch, math, os, io, shutil
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Dataset, Features, Value, Audio
from huggingface_hub import login, snapshot_download
import numpy as np
import torchaudio

# ---------------------------------------------------------------
# 1️⃣ Config
# ---------------------------------------------------------------
REPO_ID   = "palli23/icelandic-podcasts-split"
MODEL_ID  = "openai/whisper-small"  # Lighter for test (swap to your Large after)
NEW_REPO  = "palli23/icelandic-podcasts-split-whisper-test10"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BEAM_SIZE = 5
NUM_EXAMPLES = 10

# ---------------------------------------------------------------
# 2️⃣ Convert HF model to Faster-Whisper format (int8 quant)
# ---------------------------------------------------------------
print(f"🔄 Converting {MODEL_ID} to Faster-Whisper int8 format …")
CONVERTED_MODEL_DIR = Path("/content/converted_whisper_model")
if CONVERTED_MODEL_DIR.exists():
    shutil.rmtree(CONVERTED_MODEL_DIR)

# Download HF model locally
hf_model_dir = snapshot_download(MODEL_ID, local_dir="/content/hf_model")
print(f"📥 Downloaded HF model to {hf_model_dir}")

# Convert using ct2-transformers-converter
!ct2-transformers-converter \
    --model {hf_model_dir} \
    --output_dir {CONVERTED_MODEL_DIR} \
    --quantization int8_float16 \
    --force

print(f"✅ Converted model saved to {CONVERTED_MODEL_DIR}")

# ---------------------------------------------------------------
# 3️⃣ Load dataset (limit to 10)
# ---------------------------------------------------------------
print(f"⬇️  Loading dataset {REPO_ID} (first {NUM_EXAMPLES} examples) …")
dataset = load_dataset(REPO_ID)
if "train" in dataset:
    dataset = dataset["train"]
else:
    dataset = dataset[list(dataset.keys())[0]]
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))
dataset = dataset.select(range(NUM_EXAMPLES))

print(f"📁 Dataset loaded: {len(dataset)} examples.")

# ---------------------------------------------------------------
# 4️⃣ Load Faster Whisper model
# ---------------------------------------------------------------
print(f"📦 Loading Faster Whisper model from {CONVERTED_MODEL_DIR} on {DEVICE} …")
from faster_whisper import WhisperModel

model = WhisperModel(
    str(CONVERTED_MODEL_DIR),
    device=DEVICE,
    compute_type="int8_float16" if DEVICE == "cuda" else "int8",
    download_root="."
)

# ---------------------------------------------------------------
# 5️⃣ Confidence helper (robust to missing avg_logprob)
# ---------------------------------------------------------------
def compute_confidence(avg_logprob):
    if avg_logprob is None:
        return 0.0
    return round(math.exp(avg_logprob), 3)

# ---------------------------------------------------------------
# 6️⃣ Individual Inference (single for stability)
# ---------------------------------------------------------------
results = []
print("🚀 Running Faster Whisper inference …")

for idx, example in enumerate(tqdm(dataset, desc="🔊 Transcribing")):
    try:
        audio_info = example["audio"]
        # Manual decode
        if isinstance(audio_info, dict):
            if 'bytes' in audio_info:
                waveform, sr = torchaudio.load(io.BytesIO(audio_info['bytes']))
            elif 'path' in audio_info:
                waveform, sr = torchaudio.load(audio_info['path'])
            else:
                raise ValueError(f"Unsupported audio format: {audio_info.keys()}")
            audio_array = waveform.mean(0).numpy().astype(np.float32)
        else:
            audio_array = np.array(audio_info).astype(np.float32)

        if len(audio_array) == 0:
            raise ValueError("Empty audio")

        # Transcribe
        segments, info = model.transcribe(
            audio_array,
            beam_size=BEAM_SIZE,
            language="is",
            task="transcribe",
            vad_filter=True
        )
        full_text = " ".join(seg.text.strip() for seg in segments).strip()
        # Safe access to avg_logprob (fallback to None if missing)
        avg_logprob = getattr(info, 'avg_logprob', None) if info else None

        results.append({
            "example_id": example.get("id", idx),
            "text": full_text,
            "confidence": compute_confidence(avg_logprob)
        })

        # Memory management
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"⚠️  Skipped example {example.get('id', idx)}: {e}")
        results.append({"example_id": example.get("id", idx), "text": "", "confidence": None})

print(f"✅ Finished {len(results)} examples.")

# ---------------------------------------------------------------
# 7️⃣ Create and upload dataset
# ---------------------------------------------------------------
print("🧩 Creating new dataset …")
if len(results) > 0:
    features = Features({
        "example_id": Value("int32"),
        "text": Value("string"),
        "confidence": Value("float32")
    })
    ds_new = Dataset.from_list(results, features=features)
else:
    ds_new = Dataset.from_list([])

ds_new.save_to_disk("/content/icelandic_podcasts_split_whisper_test10")

print("🔐 Logging in to Hugging Face …")
login()

print(f"☁️  Uploading dataset to {NEW_REPO} …")
ds_new.push_to_hub(NEW_REPO, private=True)
print(f"✅ Upload complete → https://huggingface.co/datasets/{NEW_REPO}")

# ---------------------------------------------------------------
# 8️⃣ Quick preview
# ---------------------------------------------------------------
non_empty = sum(1 for r in results if r["text"].strip())
avg_conf = np.mean([r["confidence"] for r in results if r["confidence"] is not None]) if any(r["confidence"] is not None for r in results) else 0.0
print(f"📊 Quick Stats: {non_empty}/{len(results)} non-empty transcripts | Avg Confidence: {avg_conf:.3f}")
if results:
    print("Sample:", results[0])
