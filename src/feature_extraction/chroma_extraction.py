import datetime
import os
import librosa
import numpy as np
import pandas as pd


def extract_chroma(file_path, hop_length=512, n_fft=2048):
    """
    Extract Chroma features from an audio file.

    Parameters:
        file_path (str): Path to the .wav file.
        hop_length (int): Number of samples between successive frames.
        n_fft (int): Length of the FFT window.

    Returns:
        np.array: Extracted Chroma features or None if an error occurs.
    """
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, sr=None)
        # Extract Chroma
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sample_rate, hop_length=hop_length, n_fft=n_fft)
        # Average Chroma across time frames
        chroma_processed = np.mean(chroma.T, axis=0)

        return chroma_processed
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Update the process_all_audio_files function to call extract_chroma instead of extract_mfcc
# and to save the features with a prefix indicating they are chroma features


def process_all_audio_files(base_path):
    """
    Process all .wav files in the dataset, extracting Chroma features.

    Parameters:
        base_path (str): Base path to the dataset directories.
    """
    features = []
    error_files = []

    # Loop through all genre folders
    for subdir, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                # Extract the genre from the directory name
                genre = os.path.basename(subdir)
                # Extract Chroma features
                chroma = extract_chroma(file_path)
                if chroma is not None:
                    features.append([file] + [genre] + chroma.tolist())
                else:
                    error_files.append(file_path)

    # Save to DataFrame only if features were extracted
    if features:
        feature_columns = ['file', 'genre'] + \
            [f'chroma_{i}' for i in range(len(features[0]) - 2)]
        df = pd.DataFrame(features, columns=feature_columns)
        date_time = datetime.datetime.now().strftime("%d%m%Y_%H%M")
        df.to_csv(f'features/chroma/features_{date_time}.csv', index=False)
        print(
            f"Feature extraction complete. Processed {len(features)} files with {len(error_files)} errors.")
    else:
        print("No features were extracted.")

    if error_files:
        print(f"Errors occurred with the following files: {error_files}")
        with open('error_files.txt', 'w') as f:
            for file_path in error_files:
                f.write(f"{file_path}\n")
        print("List of problematic files saved to 'error_files.txt'.")


# Ensure the base_path correctly points to the genres_original folder within your data structure
if __name__ == '__main__':
    base_path = 'data/'
    process_all_audio_files(base_path)
