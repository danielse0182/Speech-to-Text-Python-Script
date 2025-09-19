# Speech-to-Text-Python-Script
A simple and efficient script to convert .m4a audio recordings into transcribed text using HuggingFace's speech-to-text models. Includes audio preprocessing (format conversion to 16kHz WAV) and output to CSV for easy integration with data tools. Accuracy 77%+ with the right model!
Here's a revised and detailed description with explicit instructions for **installing Python dependencies** and the exact command syntax. I've also clarified the token input step:

---

### 🎙️ **Speech-to-Text Python Script (HuggingFace + FFmpeg)**
A streamlined script to convert **.m4a audio recordings** into **transcribed text** using HuggingFace speech-to-text models. Includes audio preprocessing (convert to 16kHz WAV) and structured **CSV output**. Achieves **77%+ accuracy** with the right model!

---

#### 🔧 **Key Features:**
- **HuggingFace API Token Authentication**: Required for access to pre-trained models.
- **FFmpeg Audio Conversion**: Converts `.m4a` to 16kHz mono WAV (essential for model compatibility).
- **Flexible Python Environment**: Works with `python`, `python3`, or specific versions (e.g., `python3.11`).
- **CSV Output**: Transcriptions saved in a `.csv` file for easy analysis.

---

#### 📦 **Prerequisites:**
1. **Python Dependencies**
   Install the required Python modules using:
   ```bash
   pip install torch transformers soundfile pandas
   ```
   - **`torch`**: PyTorch for model execution.
   - **`transformers`**: HuggingFace library for speech-to-text models.
   - **`soundfile`**: For loading and processing WAV audio files.
   - **`pandas`**: For saving transcriptions to a `.csv` file.

2. **HuggingFace API Token**
   - Create an account at [https://huggingface.co/](https://huggingface.co/).
   - Generate an API token from *Profile > Settings > API Tokens*.
   - **Input the token when prompted during script execution** (if not set in the environment).

3. **FFmpeg Installation**
   - Download and install FFmpeg from [https://ffmpeg.org/](https://ffmpeg.org/).
   - Ensure `ffmpeg.exe` is in your system's PATH (Windows) or terminal (Linux/macOS).

---

#### 🚀 **Step-by-Step Usage:**
1. **Install Dependencies**
   Run the installation command in your terminal:
   ```bash
   pip install torch transformers soundfile pandas
   ```

2. **Convert .m4a to 16kHz WAV**
   Use FFmpeg to preprocess the audio file:
   ```bash
   ffmpeg -i "Path_of_your_audio_file/xxxxx.m4a" -acodec pcm_s16le -ar 16000 -ac 1 "Path_of_converted_file/xxxxx.wav"
   ```

3. **Run the Python Script**
   Execute the script with your chosen Python interpreter:
   ```bash
   python wav-to-text-large.py <input_wav> <output_csv>
   ```
   - Replace `<input_wav>` with the path to your 16kHz `.wav` file.
   - Replace `<output_csv>` with the desired `.csv` file path.
   - **During execution, the script will prompt you to input your HuggingFace API Token** if it’s not pre-configured in the environment.

---

#### 📤 **Output:**
- A `.csv` file containing transcribed text (e.g., timestamps and recognized sentences).

---

#### ⚙️ **Notes:**
- **Model Accuracy**: Achieves **77%+ accuracy** using HuggingFace's pre-trained models (e.g., Whisper or Wav2Vec2).
- **GPU Support**: For faster inference, use a CUDA-enabled GPU (PyTorch handles this automatically).
- **API Token Input**: If the token isn’t set in the environment, the script will prompt you to enter it at runtime.

---

#### 📌 **Example Command:**
```bash
# Step 1: Convert .m4a to 16kHz WAV
ffmpeg -i "audio/recording.m4a" -acodec pcm_s16le -ar 16000 -ac 1 "output/recording.wav"

# Step 2: Install Python dependencies
pip install torch transformers soundfile pandas

# Step 3: Run the script
python wav-to-text-large.py output/recording.wav results.csv
```

---

#### 📚 **Supported Models:**
- Uses HuggingFace's [AutoModelForCTC](https://huggingface.co/docs/transformers/v4.32.0/en/model_doc/ctc) and [Wav2Vec2](https://huggingface.co/fairseq) for speech recognition.

---

**Perfect for developers, researchers, or anyone needing to process audio data into text using open-source tools. 🧠✨**

---

### ✅ **Key Additions:**
- **Explicit Python dependencies** with installation commands (`pip install torch transformers soundfile pandas`).
- **Clear step in the example** showing how to install all required modules.
- **Token input clarification** during script runtime if not pre-configured.

---

THANK YOU AND GOD BLESS YOU ALWAYS! 😊
