# To-Do List

This is a list of planned features and improvements for the audio transcription application.

## High Priority

-   [ ] **Refactor Speaker Input**: Remove the hardcoded number of speakers and find a way to pass it as a command-line argument to make the script more flexible.
-   [ ] **Error Handling**: Improve error handling for cases like invalid audio file formats or failed downloads of models.
-   [ ] **Configuration File**: Add support for a configuration file (e.g., `config.yaml`) to manage settings like the Whisper model, Hugging Face token, and other parameters, so they don't need to be specified on the command line every time.

## Medium Priority

-   [ ] **Automatic Speaker Detection**: Explore `pyannote.audio`'s capabilities to automatically detect the number of speakers, removing the need for manual input.
-   [ ] **Alternative Diarization Models**: Add support for other speaker diarization models to offer more choices to the user.
-   [ ] **Output Formats**: Allow the user to specify the output format (e.g., plain text, JSON, SRT subtitles).
-   [ ] **Progress Bar**: Implement a more detailed progress bar, especially for the transcription process, as it can be time-consuming.

## Low Priority

-   [ ] **Real-time Transcription**: Investigate the feasibility of real-time (streaming) transcription and diarization.
-   [ ] **GUI Interface**: Create a simple graphical user interface (GUI) for users who are not comfortable with the command line.
-   [ ] **Dockerization**: Package the application in a Docker container to simplify the setup and dependency management.
