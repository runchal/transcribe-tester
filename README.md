# Audio Transcription with Speaker Diarization

This is a command-line tool to transcribe audio files and attribute the speech to different speakers. It uses OpenAI's Whisper for highly accurate speech-to-text transcription and `pyannote.audio` for speaker diarization.

## Features

-   **Accurate Transcription**: Leverages Whisper models for state-of-the-art results.
-   **Speaker Diarization**: Identifies who spoke and when, even with multiple speakers.
-   **Timestamped Output**: Produces a text file with speaker labels and timestamps for each segment of speech.
-   **Configurable Models**: Allows choosing different Whisper model sizes to balance speed and accuracy.

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```
*(Assuming this project will be hosted in a Git repository)*

### 2. System Dependencies (macOS)

This application has system-level dependencies. The easiest way to install them on macOS is with [Homebrew](https://brew.sh/).

```bash
# Install core dependencies
brew install sentencepiece protobuf

# You also need the Xcode Command Line Tools
xcode-select --install
```

### 3. Python Dependencies

Install the required Python packages using `pip`.

```bash
pip install -r requirements.txt
```

## Authentication

This tool uses the `pyannote/speaker-diarization-3.1` model, which requires a Hugging Face authentication token.

1.  Create a Hugging Face account at <https://huggingface.co/>.
2.  Generate an access token from your settings: <https://huggingface.co/settings/tokens>.
3.  Log in via the command line. You will be prompted to paste your token.

    ```bash
    huggingface-cli login
    ```

## Usage

Run the script from your terminal, providing the path to your audio file.

```bash
python transcribe.py <path_to_audio_file> [--model <model_name>]
```

### Arguments

-   `audio_path`: (Required) The full path to the audio file you want to transcribe (e.g., `sample-files/audio.m4a`).
-   `--model`: (Optional) The Whisper model to use. Defaults to `base`. Other options include `tiny`, `small`, `medium`, and `large`. Larger models are more accurate but slower and require more memory.

### Example

```bash
python transcribe.py "sample files/Therapy 2025.05.27.m4a" --model medium
```

This will generate a file named `Therapy 2025.05.27_transcript.txt` with content like this:

```
[00:00:12] SPEAKER_01: Hello, this is the first speaker.
[00:00:15] SPEAKER_02: And this is the second speaker responding.
...
```
