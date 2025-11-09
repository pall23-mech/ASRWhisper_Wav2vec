# ==============================================================
# 🍬 Whisper inference for Icelandic podcasts (10 samples, repeat penalty + logprob confidence)
# ==============================================================
#successful 8-bit inference with confidence on 10 samples
!pip install -q torch torchaudio transformers datasets huggingface_hub accelerate tqdm soundfile huggingface_hub[hf_transfer]

import torch, math, os, io
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np
import torchaudio

# ---------------------------------------------------------------
# 1️⃣ Config
# ---------------------------------------------------------------
REPO_ID   = "palli23/icelandic-podcasts-split"
MODEL_ID  = "palli23/whisper-large-icelandic-multi-aug-v2"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EXAMPLES = 10
REPETITION_PENALTY = 1.2  # Mild anti-repeat (tune 1.1-1.5)

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
# 5️⃣ Inference & Print Each (direct generate with repeat penalty)
# ---------------------------------------------------------------
print("🚀 Running Whisper inference with repeat penalty & printing each …")

for idx, example in enumerate(tqdm(list(dataset.take(NUM_EXAMPLES)), desc="🔊 Transcribing")):
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

        # Print each result
        print(f"\n--- Example {idx} (ID: {example.get('id', idx)}) ---")
        print(f"Text: {text}")
        print(f"Confidence: {confidence:.3f}")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n--- Example {idx} (ID: {example.get('id', idx)}) ---")
        print(f"Error: {e}")
        print(f"Text: ''")
        print(f"Confidence: 0.000")

print("✅ Finished all 10 examples.")
