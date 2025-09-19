# wav-to-text-largenew.py
import os
import re
import torch
import pandas as pd
from pyannote.audio import Pipeline
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
import torchaudio
import logging
import warnings
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime

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
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)  # Fallback for older versions

        # Initialize diarization pipeline
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=self.hf_token
        )
        
        # Try to move pipeline to GPU if supported
        try:
            self.diarization_pipeline.to(self.device)
        except AttributeError:
            logger.warning("⚠️ Pyannote pipeline doesn't support device movement - will use CPU for diarization")

        # Whisper processor setup
        self.model_size = "large"
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{self.model_size}")
        self.tokenizer = WhisperTokenizer.from_pretrained(
            f"openai/whisper-{self.model_size}",
            language=language,
            task="transcribe"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            f"openai/whisper-{self.model_size}"
        ).to(self.device)
        
        # Enable mixed precision if using GPU
        if self.device.type == 'cuda':
            self.model.half()
        
        self.model.eval()

    def load_audio(self, audio_path):
        """Load and validate audio file"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"❌ Audio file not found: {audio_path}")
        
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sampling_rate:
            raise ValueError(f"❌ Audio must be {self.sampling_rate}Hz")
        return waveform, sample_rate

    def extract_speaker_number(self, label: str) -> int:
        """Extract speaker number from label (robust for all formats)"""
        match = re.search(r'\d+', label)
        if not match:
            raise ValueError(f"❌ Could not extract speaker number from label: {label}")
        return int(match.group())

    def diarize_and_transcribe(self, audio_path, output_path):
        """Full diarization + transcription workflow"""
        try:
            waveform, sample_rate = self.load_audio(audio_path)
            diarization = self.diarization_pipeline(audio_path)
            
            # Batch preparation
            segments_audio = []
            segment_info = []
            
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                start = segment.start
                end = segment.end
                segment_audio = waveform[:, int(start*sample_rate):int(end*sample_rate)]
                segments_audio.append(segment_audio[0].numpy())
                segment_info.append((start, end, speaker))
            
            # Batch feature extraction with padding
            inputs = self.feature_extractor(
                segments_audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Move to GPU and convert to FP16
            inputs = {key: val.to(self.device).half() if self.device.type == 'cuda' else val.to(self.device) 
                     for key, val in inputs.items()}
            
            # Batched inference using inference mode
            with torch.inference_mode():
                predicted_ids = self.model.generate(
                    inputs["input_features"],
                    attention_mask=inputs.get("attention_mask"),
                    task="transcribe",
                    language=self.language,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    length_penalty=1.2,
                    early_stopping=True,
                    temperature=0.0,
                    max_length=448
                )
            
            # Process results
            transcriptions = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            results = []
            for (start, end, speaker), transcription in zip(segment_info, transcriptions):
                results.append({
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "speaker": f"Speaker {self.extract_speaker_number(speaker)}",
                    "transcription": self._restore_punctuation(transcription.strip())
                })

            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            return df

        except Exception as e:
            logger.error("Error during processing", exc_info=True)
            raise RuntimeError(f"❌ Processing failed: {e}")

    def _restore_punctuation(self, text):
        """Restore punctuation using NLTK for better readability"""
        try:
            # Try with punkt_tab first (newer method)
            sentences = sent_tokenize(text, language='indonesian')
        except LookupError:
            # Fallback to punkt if punkt_tab fails
            nltk.download('punkt', quiet=True)
            sentences = sent_tokenize(text)
        return ' '.join(sentences)

if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) != 3:
        print("Usage: python wav-to-text-largenew.py <input_wav> <output_csv>")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        print("CUDA available:", torch.cuda.is_available())
        print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
        
        print("ℹ️ Loading models...")
        start_time = time.time()
        
        # Initialize processor with Indonesian language
        processor = SpeechToTextProcessor(language='id')
        print(f"✅ Models loaded in {time.time()-start_time:.1f}s. Processing audio...")
        
        # Add timestamp to output file for versioning
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}-{timestamp}{ext}"
        
        result_df = processor.diarize_and_transcribe(audio_path, output_path)
        print(f"✅ Transcription saved to {output_path}")
        print("First 5 rows:")
        print(result_df.head())

    except Exception as e:
        print(f"❌ Final Error: {e}")
        sys.exit(1)
