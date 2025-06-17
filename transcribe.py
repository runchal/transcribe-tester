import os
import sys
import argparse
import whisper
import torch
from huggingface_hub import HfFolder
from pyannote.audio import Pipeline

def format_time(seconds):
    """Converts seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    """
    Main function to run the transcription and diarization process.
    """
    parser = argparse.ArgumentParser(description="Transcribe an audio file with speaker diarization.")
    parser.add_argument("audio_path", type=str, help="Path to the audio file to transcribe.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model to use (e.g., tiny, base, small, medium, large).")
    args = parser.parse_args()

    audio_path = args.audio_path
    whisper_model_name = args.model

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        sys.exit(1)

    # 1. Get number of speakers from user
    while True:
        try:
            num_speakers_str = input("Enter the number of speakers in the audio file: ")
            num_speakers = int(num_speakers_str)
            if num_speakers > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # 2. Check for Hugging Face token
    if HfFolder.get_token() is None:
        print("Hugging Face token not found.")
        print("Please log in using 'huggingface-cli login'.")
        print("You can get a token from https://huggingface.co/settings/tokens")
        sys.exit(1)

    # 3. Set up models and pipelines
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading speaker diarization pipeline...")
    # Using the latest pyannote pipeline
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=True
    ).to(device)

    print("Loading Whisper model...")
    whisper_model = whisper.load_model(whisper_model_name, device=device)

    # 4. Perform Diarization
    print("Performing speaker diarization...")
    diarization = diarization_pipeline(audio_path, num_speakers=num_speakers)

    # 5. Perform Transcription
    print("Transcribing audio file...")
    transcription_result = whisper_model.transcribe(audio_path, word_timestamps=True)
    words = transcription_result['words']

    # 6. Align transcription with diarization
    print("Aligning transcription with speakers...")
    word_idx = 0
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_time, end_time = segment.start, segment.end
        
        # Find words that fall within the speaker's segment
        while word_idx < len(words) and words[word_idx]['start'] < end_time:
            if words[word_idx]['start'] >= start_time:
                words[word_idx]['speaker'] = speaker
            word_idx += 1
    
    # 7. Format the output
    output_segments = []
    if words:
        current_speaker = words[0].get('speaker', 'UNKNOWN')
        current_text = ""
        segment_start_time = words[0]['start']

        for word in words:
            speaker = word.get('speaker', 'UNKNOWN')
            if speaker != current_speaker:
                # New speaker, finalize previous segment
                output_segments.append(f"[{format_time(segment_start_time)}] {current_speaker}: {current_text.strip()}")
                # Start new segment
                current_speaker = speaker
                current_text = word['text']
                segment_start_time = word['start']
            else:
                current_text += " " + word['text']
        
        # Add the last segment
        output_segments.append(f"[{format_time(segment_start_time)}] {current_speaker}: {current_text.strip()}")

    # 8. Write to output file
    output_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_transcript.txt"
    with open(output_filename, "w") as f:
        f.write("\n".join(output_segments))

    print(f"\nTranscription complete. Output saved to '{output_filename}'")

if __name__ == "__main__":
    main()
