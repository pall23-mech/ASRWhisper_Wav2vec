# ==============================================================
# 🍬 Whisper inference for Icelandic podcasts (10 samples, 8-bit, repeat penalty + timing/confidence)
# ==============================================================

!pip install -q torch torchaudio transformers datasets huggingface_hub accelerate tqdm soundfile huggingface_hub[hf_transfer]
!pip install -U bitsandbytes

import time
import torch, math, os, io
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, BitsAndBytesConfig
import numpy as np
import torchaudio

# ---------------------------------------------------------------
# 1️⃣ Config
# ---------------------------------------------------------------
REPO_ID   = "palli23/icelandic-podcasts-split"
MODEL_ID  = "palli23/whisper-large-icelandic-multi-aug-v2"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EXAMPLES = 10
REPETITION_PENALTY = 1.2  # Mild anti-repeat

# ---------------------------------------------------------------
# 2️⃣ Load dataset (streaming, take 10, decode=False)
# ---------------------------------------------------------------
print(f"⬇️  Loading dataset {REPO_ID} (streaming, first {NUM_EXAMPLES}) …")
dataset = load_dataset(REPO_ID, streaming=True)
if "train" in dataset:
    dataset = dataset["train"]
else:
    dataset = dataset[list(dataset.keys())[0]]
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

print(f"📁 Dataset loaded (streaming mode).")

# ---------------------------------------------------------------
# 3️⃣ Load 8-bit Quantized Whisper model
# ---------------------------------------------------------------
print(f"📦 Loading 8-bit quantized Whisper model {MODEL_ID} on {DEVICE} …")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 8-bit config
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto"
)

# ---------------------------------------------------------------
# 4️⃣ Confidence from logprobs (mean exp(max logp per token, skip BOS/pad)
# ---------------------------------------------------------------
def compute_confidence(logits):
    if logits is None or len(logits) == 0:
        return 0.0
    logprobs = []
    for l in logits[1:]:  # Skip first token
        logp = torch.log_softmax(l, dim=-1).max(dim=-1)[0]
        logprobs.extend(logp.tolist())
    if not logprobs:
        return 0.0
    mean_logprob = sum(logprobs) / len(logprobs)
    return round(math.exp(mean_logprob), 3)

# ---------------------------------------------------------------
# 5️⃣ Inference & Print Each (direct generate with repeat penalty + timing)
# ---------------------------------------------------------------
overall_start = time.time()
example_times = []

print("🚀 Running Whisper inference with repeat penalty & printing each …")

for idx, example in enumerate(tqdm(list(dataset.take(NUM_EXAMPLES)), desc="🔊 Transcribing")):
    start_time = time.time()
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
            audio_array = waveform.mean(0).numpy().astype(np.float16)  # fp16 for 8-bit model
        else:
            audio_array = np.array(audio_info).astype(np.float16)

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
                num_beams=1,  # Greedy for simplicity
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

        example_time = time.time() - start_time
        example_times.append(example_time)

        # Print each result
        print(f"\n--- Example {idx} (ID: {example.get('id', idx)}) ---")
        print(f"Text: {text}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Time: {example_time:.3f}s")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        example_time = time.time() - start_time
        example_times.append(example_time)
        print(f"\n--- Example {idx} (ID: {example.get('id', idx)}) ---")
        print(f"Error: {e}")
        print(f"Text: ''")
        print(f"Confidence: 0.000")
        print(f"Time: {example_time:.3f}s")

total_time = time.time() - overall_start
mean_confidence = np.mean([c for c in [r['confidence'] for r in results] if c > 0]) if any(c > 0 for r in results) else 0.0
mean_time = np.mean(example_times)

print(f"\n✅ Completed all {NUM_EXAMPLES} examples.")
print(f"Total time: {total_time:.3f}s ({total_time / 60:.1f} min)")
print(f"Mean confidence: {mean_confidence:.3f}")
print(f"Mean time per example: {mean_time:.3f}s")

# ---------------------------------------------------------------
# 6️⃣ Create and upload dataset
# ---------------------------------------------------------------
print("🧩 Creating new dataset …")
results = []
for idx, example in enumerate(tqdm(list(dataset.take(NUM_EXAMPLES)), desc="🔊 Transcribing")):
    start_time = time.time()
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
            audio_array = waveform.mean(0).numpy().astype(np.float16)
        else:
            audio_array = np.array(audio_info).astype(np.float16)

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
                num_beams=1,
                language="is",
                task="transcribe",
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                repetition_penalty=REPETITION_PENALTY
            )

        # Decode text
        text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0].strip()

        # Extract logprobs
        logits = generated_ids.scores
        confidence = compute_confidence(logits)

        example_time = time.time() - start_time
        example_times.append(example_time)

        results.append({
            "example_id": example.get("id", idx),
            "text": text,
            "confidence": confidence
        })

        # Print each result
        print(f"\n--- Example {idx} (ID: {example.get('id', idx)}) ---")
        print(f"Text: {text}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Time: {example_time:.3f}s")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        example_time = time.time() - start_time
        example_times.append(example_time)
        print(f"\n--- Example {idx} (ID: {example.get('id', idx)}) ---")
        print(f"Error: {e}")
        print(f"Text: ''")
        print(f"Confidence: 0.000")
        print(f"Time: {example_time:.3f}s")

total_time = time.time() - overall_start
mean_confidence = np.mean([c for c in [r['confidence'] for r in results] if c > 0]) if any(c > 0 for r in results) else 0.0
mean_time = np.mean(example_times)

print(f"\n✅ Completed all {NUM_EXAMPLES} examples.")
print(f"Total time: {total_time:.3f}s ({total_time / 60:.1f} min)")
print(f"Mean confidence: {mean_confidence:.3f}")
print(f"Mean time per example: {mean_time:.3f}s")

# ---------------------------------------------------------------
# 6️⃣ Create and upload dataset
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
