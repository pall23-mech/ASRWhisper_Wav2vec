# ==============================================================
# 🧠 Whisper inference for Icelandic podcasts (full dataset)
# - Prevents truncation via forced decoder IDs + beam search
# - Adds confidence column
# - Prints every 50th example
# - Uploads to Hugging Face
# ==============================================================

!pip install -q torch torchaudio transformers datasets huggingface_hub accelerate tqdm soundfile huggingface_hub[hf_transfer]

import torch, math, io, gc
import numpy as np
import torchaudio
from tqdm import tqdm
from datasets import load_dataset, Audio, Dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import login

# -------------------- Config --------------------
REPO_ID   = "palli23/icelandic-podcasts-split"
MODEL_ID  = "palli23/whisper-large-icelandic-multi-aug-v2"
NEW_REPO  = "palli23/icelandic-podcasts-split-whisper"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
REPETITION_PENALTY = 1.1       # lowered slightly
MAX_LENGTH = 768               # safe for up to ~50s audio
NUM_BEAMS = 3                  # small beam search for robustness

# -------------------- HF login --------------------
print("🔐 Logging in to Hugging Face …")
login()  # will prompt for your token

# -------------------- Load dataset --------------------
print(f"📦 Loading dataset {REPO_ID} …")
dataset = load_dataset(REPO_ID, split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))
print(f"🎧 Total clips: {len(dataset)}")

# -------------------- Load model --------------------
print(f"📦 Loading Whisper model {MODEL_ID} on {DEVICE} …")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# -------------------- Confidence helper --------------------
def compute_confidence(logits):
    """Estimate confidence from token logprobs."""
    if logits is None or len(logits) == 0:
        return 0.0
    logprobs = []
    for l in logits[1:]:  # skip BOS
        logp = torch.log_softmax(l, dim=-1).max(dim=-1)[0]
        logprobs.extend(logp.tolist())
    if not logprobs:
        return 0.0
    mean_logprob = sum(logprobs) / len(logprobs)
    return round(math.exp(mean_logprob), 3)

# -------------------- Inference --------------------
results = []
print("\n🚀 Starting inference …\n")

forced_decoder_ids = processor.get_decoder_prompt_ids(language="is", task="transcribe")

for idx, example in enumerate(tqdm(dataset, desc="🔊 Transcribing")):
    try:
        audio_info = example["audio"]

        # Decode audio safely
        if isinstance(audio_info, dict):
            if "bytes" in audio_info:
                waveform, sr = torchaudio.load(io.BytesIO(audio_info["bytes"]))
            elif "path" in audio_info:
                waveform, sr = torchaudio.load(audio_info["path"])
            else:
                raise ValueError(f"Unsupported audio format: {audio_info.keys()}")
            audio_array = waveform.mean(0).numpy()
        else:
            audio_array = np.array(audio_info)

        if len(audio_array) == 0:
            raise ValueError("Empty audio")

        # optional silence trim to reduce early stop
        try:
            audio_array = torchaudio.transforms.Vad(sample_rate=16000)(torch.tensor(audio_array)).numpy()
        except Exception:
            pass

        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                inputs.input_features,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                language="is",
                task="transcribe",
                repetition_penalty=REPETITION_PENALTY,
                forced_decoder_ids=forced_decoder_ids,  # ✅ prevents truncation
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode and compute confidence
        text = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
        confidence = compute_confidence(generated.scores)

        results.append({
            "audio": example["audio"],
            "name": example["name"],
            "text": text,
            "confidence": confidence,
        })

        # print every 50th example
        if idx % 50 == 0:
            print(f"\n--- Example {idx} ---")
            print(f"🗣️  Text: {text[:200]}")
            print(f"📊 Confidence: {confidence:.3f}")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"⚠️  Error at index {idx}: {e}")
        results.append({
            "audio": example.get("audio", None),
            "name": example.get("name", ""),
            "text": "",
            "confidence": 0.0,
        })
        continue

print("\n✅ Inference complete — building new dataset …")

# -------------------- Create new dataset --------------------
new_ds = Dataset.from_list(results)
print(new_ds)

# -------------------- Save + upload --------------------
LOCAL_DIR = "icelandic_podcasts_split_whisper"
new_ds.save_to_disk(LOCAL_DIR)

print("☁️  Uploading dataset to Hugging Face …")
new_ds.push_to_hub(NEW_REPO, private=True)

print(f"✅ Upload complete → https://huggingface.co/datasets/{NEW_REPO}")
