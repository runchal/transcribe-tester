import os
import sys
import argparse
import subprocess
from faster_whisper import WhisperModel
import torch
from huggingface_hub import HfFolder
from pyannote.audio import Pipeline

def convert_to_wav(audio_path):
    """Converts an audio file to WAV format using ffmpeg."""
    temp_wav_path = os.path.splitext(audio_path)[0] + ".wav"
    command = [
        "ffmpeg",
        "-i", audio_path,
        "-ar", "16000",        # Resample to 16kHz
        "-ac", "1",            # Convert to mono
        "-c:a", "pcm_s16le",  # Use standard WAV codec
        "-y",                 # Overwrite output file if it exists
        temp_wav_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully converted {audio_path} to {temp_wav_path}")
        return temp_wav_path, True
    except subprocess.CalledProcessError as e:
        print(f"Error converting file to WAV: {e.stderr}")
        return audio_path, False

def main():
    """
    Main function to run the transcription and diarization process.
    """
    parser = argparse.ArgumentParser(description="Transcribe an audio file with speaker diarization.")
    parser.add_argument("audio_path", type=str, help="Path to the audio file to transcribe.")
    parser.add_argument("--model", type=str, default="large-v3", help="Whisper model to use.")
    parser.add_argument("--speaker-names", type=str, nargs='+', help="Names for the speakers.")
    args = parser.parse_args()

    original_audio_path = args.audio_path
    if not os.path.exists(original_audio_path):
        print(f"Error: Audio file not found at '{original_audio_path}'")
        sys.exit(1)

    # --- Audio Conversion Step ---
    is_temp_file = False
    if not original_audio_path.lower().endswith('.wav'):
        print("Input file is not in WAV format. Converting using ffmpeg...")
        audio_to_process, is_temp_file = convert_to_wav(original_audio_path)
        if not is_temp_file:
            sys.exit(1) # Exit if conversion failed
    else:
        audio_to_process = original_audio_path
    # --- End Conversion Step ---
    
    try:
        # 1. Set number of speakers
        num_speakers = 2

        # 2. Check for Hugging Face token
        if HfFolder.get_token() is None:
            print("Hugging Face token not found. Please log in using 'huggingface-cli login'.")
            sys.exit(1)

        # 3. Set up models and pipelines
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"
        
        print(f"Using device: {device} and compute type: {compute_type}")

        print("Loading speaker diarization pipeline...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=True).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        print("Loading Whisper model...")
        whisper_model = WhisperModel(args.model, device=device, compute_type=compute_type)

        # 4. Perform Diarization
        print("Performing speaker diarization...")
        diarization = diarization_pipeline(audio_to_process, num_speakers=num_speakers)

        # 5. Perform Transcription
        print("Transcribing audio file...")
        segments, _ = whisper_model.transcribe(audio_to_process, word_timestamps=True)
        
        words = []
        for segment in segments:
            for word in segment.words:
                words.append({'text': word.word, 'start': word.start, 'end': word.end})

        # 6. Align transcription with diarization
        print("Aligning transcription with speakers...")
        word_idx = 0
        unique_speakers = diarization.labels()

        for segment, _, speaker_label in diarization.itertracks(yield_label=True):
            start_time, end_time = segment.start, segment.end
            
            while word_idx < len(words) and words[word_idx]['start'] < end_time:
                if words[word_idx]['start'] >= start_time:
                    words[word_idx]['speaker'] = speaker_label
                word_idx += 1
        
        # 7. Format the output
        output_segments = []
        if words:
            speaker_map = {}
            if args.speaker_names:
                for i, speaker_label in enumerate(unique_speakers):
                    speaker_map[speaker_label] = args.speaker_names[i] if i < len(args.speaker_names) else speaker_label
            
            if 'speaker' not in words[0]:
                words[0]['speaker'] = 'UNKNOWN'

            current_speaker_label = words[0]['speaker']
            current_text = ""

            for word in words:
                speaker_label = word.get('speaker', 'UNKNOWN')
                if speaker_label != current_speaker_label:
                    speaker_name = speaker_map.get(current_speaker_label, current_speaker_label)
                    output_segments.append(f"{speaker_name}:")
                    output_segments.append(current_text.strip())
                    output_segments.append("")
                    current_speaker_label = speaker_label
                    current_text = word['text']
                else:
                    current_text += " " + word['text']
            
            speaker_name = speaker_map.get(current_speaker_label, current_speaker_label)
            output_segments.append(f"{speaker_name}:")
            output_segments.append(current_text.strip())

        # 8. Write to output file
        output_filename = os.path.splitext(os.path.basename(original_audio_path))[0] + "_transcript.txt"
        with open(output_filename, "w") as f:
            f.write("\n".join(output_segments))

        print(f"\nTranscription complete. Output saved to '{output_filename}'")
    finally:
        if is_temp_file:
            print(f"Cleaning up temporary file: {audio_to_process}")
            os.remove(audio_to_process)

if __name__ == "__main__":
    main()