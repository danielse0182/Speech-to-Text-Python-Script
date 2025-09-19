# wav-to-text-large-optimized.py
import os
import torch
import torchaudio
import pandas as pd
import logging
import warnings
import re
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime
import numpy as np
from scipy.signal import resample
from torchaudio.functional import highpass_biquad, lowpass_biquad
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer

# Suppress deprecated torchaudio warnings
warnings.filterwarnings("ignore", message="list_audio_backends", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToTextProcessor:
    def __init__(self, language='id'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = 16000
        self.language = language

        # Load Hugging Face token
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise EnvironmentError("❌ HUGGINGFACE_TOKEN not set")

        # Download required NLTK resources
        nltk.download('punkt', quiet=True)

        # Use official Whisper large-v3 model with Indonesian language parameter
        self.model_size = "large-v3"
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            f"openai/whisper-{self.model_size}"
        )
        self.tokenizer = WhisperTokenizer.from_pretrained(
            f"openai/whisper-{self.model_size}",
            language=self.language,
            task="transcribe"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            f"openai/whisper-{self.model_size}"
        ).to(self.device)
        self.model.eval()

    def load_audio(self, audio_path):
        """Load and validate audio file with automatic resampling"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"❌ Audio file not found: {audio_path}")
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=self.sampling_rate)
            sample_rate = self.sampling_rate
            
        return waveform, sample_rate

    def _preprocess_audio(self, waveform, sample_rate):
        """Enhance audio quality with noise reduction and normalization"""
        # Convert to numpy array
        audio_array = waveform[0].numpy()
        
        # Normalize audio
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        # Apply high-pass and low-pass filters to enhance speech frequencies
        audio_tensor = torch.tensor(audio_array).unsqueeze(0)
        audio_tensor = highpass_biquad(audio_tensor, sample_rate=sample_rate, cutoff_freq=200)
        audio_tensor = lowpass_biquad(audio_tensor, sample_rate=sample_rate, cutoff_freq=7500)
        
        return audio_tensor

    def _restore_punctuation(self, text):
        """Restore punctuation using NLTK with Indonesian language support"""
        try:
            nltk.download('punkt', quiet=True)
            sentences = sent_tokenize(text, language='indonesian')
        except LookupError:
            # Fallback to basic punctuation if NLTK fails
            sentences = [s.strip() + '.' if not s.endswith(('.', '!', '?')) else s.strip() 
                        for s in re.split(r'(?<=[.!?])\s+', text)]
        return ' '.join(sentences)

    def transcribe(self, audio_path, output_path):
        """Transcribe audio in overlapping chunks for better accuracy"""
        try:
            # Load and preprocess audio
            waveform, sample_rate = self.load_audio(audio_path)
            audio_tensor = self._preprocess_audio(waveform, sample_rate)
            
            # Split audio into overlapping segments (30s per segment, 5s overlap)
            chunk_length = 30 * sample_rate  # 30 seconds per chunk
            overlap = 5 * sample_rate  # 5 seconds overlap
            total_samples = audio_tensor.shape[1]
            results = []

            for i in range(0, total_samples, chunk_length - overlap):
                start = i / sample_rate
                end = min(i + chunk_length, total_samples) / sample_rate
                
                # Extract chunk
                chunk = audio_tensor[:, i:i + chunk_length]
                
                # Process with feature extractor
                inputs = self.feature_extractor(
                    chunk[0].numpy(),
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                )
                inputs = {key: val.to(self.device) for key, val in inputs.items()}

                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        inputs["input_features"],
                        attention_mask=inputs.get("attention_mask"),  # Use attention mask
                        task="transcribe",
                        language=self.language,
                        num_beams=15,
                        no_repeat_ngram_size=5,
                        length_penalty=1.8,
                        early_stopping=True,
                        temperature=0.0,
                        max_length=448,
                        min_length=20
                    )
                transcription = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                transcription = self._restore_punctuation(transcription)
                transcription = transcription.strip()
                
                results.append({
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "transcription": transcription
                })

            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            return df

        except Exception as e:
            logger.error("Error during processing", exc_info=True)
            raise RuntimeError(f"❌ Processing failed: {e}")

if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) != 3:
        print("Usage: python wav-to-text-large.py <input_wav> <output_csv>")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        print("ℹ️ Loading models...")
        start_time = time.time()
        
        # Initialize processor with Indonesian language
        processor = SpeechToTextProcessor(language='id')
        print(f"✅ Models loaded in {time.time()-start_time:.1f}s. Processing audio...")
        
        # Add timestamp to output file for versioning
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}-{timestamp}{ext}"
        
        result_df = processor.transcribe(audio_path, output_path)
        print(f"✅ Transcription saved to {output_path}")
        print("First 5 rows:")
        print(result_df.head())

    except Exception as e:
        print(f"❌ Final Error: {e}")
        sys.exit(1)
