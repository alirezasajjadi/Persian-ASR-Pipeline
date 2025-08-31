# Persian ASR Pipeline 🎙️

A comprehensive Persian (Farsi) Automatic Speech Recognition (ASR) pipeline that includes audio preprocessing, speaker diarization, multiple ASR model evaluation, text post-processing, intent classification, and text-to-speech synthesis.

## 🚀 Features

- **Audio Preprocessing**: Noise reduction for cleaner audio input
- **Speaker Diarization**: Automatic speaker identification and segmentation
- **Multiple ASR Models**: Compare and evaluate 5 different Persian ASR models
- **Text Post-Processing**: LLM-based text correction for improved accuracy
- **Intent Classification**: Classify user intent using BERT and LLM approaches
- **Text-to-Speech**: Convert processed text back to speech in Persian

## 📋 Requirements

### Core Dependencies
```bash
pip install -U openai-whisper
pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio
pip install torchaudio
pip install noisereduce librosa soundfile
pip install pyannote.audio
pip install faster-whisper
pip install edge-tts
```

### Required Tokens
- HuggingFace Token (for accessing models and speaker diarization)

## 🏗️ Pipeline Architecture

### 1. Audio Preprocessing
- **Noise Reduction**: Uses `noisereduce` library to clean audio files
- **Format Support**: Handles various audio formats (WAV, MP3, etc.)

### 2. Speaker Diarization
- **Model**: `pyannote/speaker-diarization-3.1`
- **Output**: Separates different speakers into individual audio files
- **Format**: Generates RTTM (Rich Transcription Time Marked) files

### 3. ASR Models Comparison

#### Model 1: Wav2Vec2 Persian V3
- **Model**: `m3hrdadfi/wav2vec2-large-xlsr-persian-v3`
- **Type**: Fine-tuned Wav2Vec2 on Persian
- **Processing**: Batch processing with custom preprocessing

#### Model 2: XLS-R Persian
- **Model**: `ghofrani/xls-r-1b-fa-cv8`
- **Type**: Fine-tuned XLS-R on Persian Common Voice 8

#### Model 3: Wav2Vec2 XLSR-53 Persian
- **Model**: `jonatasgrosman/wav2vec2-large-xlsr-53-persian`
- **Type**: Cross-lingual Speech Representations

#### Model 4: Faster Whisper
- **Implementation**: `faster-whisper` library
- **Features**: GPU acceleration, VAD filtering
- **Model Size**: Turbo (configurable)

#### Model 5: Whisper Large V3 Turbo
- **Model**: `openai/whisper-large-v3-turbo`
- **Features**: SDPA attention, timestamp generation
- **Optimization**: High precision matrix multiplication

### 4. Text Post-Processing
- **LLM Model**: `google/gemma-3n-e4b-it`
- **Function**: Corrects spelling and grammatical errors in Persian text
- **Approach**: Prompt-based correction using generative AI

### 5. Intent Classification

#### BERT-based Classification
- **Model**: `bert-base-uncased`
- **Method**: Zero-shot classification
- **Categories**: Customizable intent categories

#### LLM-based Classification
- **Model**: `google/gemma-3n-e4b-it`
- **Method**: Prompt-based classification
- **Language**: Native Persian prompts

### 6. Text-to-Speech (TTS)
- **Engine**: Microsoft Edge TTS
- **Voice**: `fa-IR-FaridNeural`
- **Output**: High-quality MP3 audio files


## 📊 Model Performance Comparison

The notebook allows you to compare different ASR models on the same audio input:

| Model | Type | Speed | Accuracy | GPU Memory |
|-------|------|-------|----------|------------|
| Wav2Vec2 Persian V3 | Fine-tuned | Medium | High | Medium |
| XLS-R Persian | Fine-tuned | Fast | High | Low |
| Wav2Vec2 XLSR-53 | Cross-lingual | Medium | Medium | Medium |
| Faster Whisper | Optimized | Very Fast | High | Low |
| Whisper Large V3 Turbo | Foundation | Fast | Very High | High |


**Note**: Make sure to replace `"your_huggingface_token"` with your actual HuggingFace token before running the code.