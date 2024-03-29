import datetime
import os
import librosa
import numpy as np
import pandas as pd


def extract_mfcc(file_path, n_mfcc=13, hop_length=512, n_fft=2048):
    """
    Extract MFCC features from an audio file.

    Parameters:
        file_path (str): Path to the .wav file.
        n_mfcc (int): Number of MFCCs to return.
        hop_length (int): Number of samples between successive frames.
        n_fft (int): Length of the FFT window.

    Returns:
        np.array: Extracted MFCC features or None if an error occurs.
    """
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, sr=None)
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        # Average MFCCs across time frames
        mfccs_processed = np.mean(mfccs.T, axis=0)

        return mfccs_processed
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def process_all_audio_files(base_path):
    """
    Process all .wav files in the dataset, extracting MFCC features.

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
                # Extract MFCCs
                mfccs = extract_mfcc(file_path)
                if mfccs is not None:
                    features.append([file] + [genre] + mfccs.tolist())
                else:
                    error_files.append(file_path)

    # Save to DataFrame
    if features:
        feature_names = ['filename', 'genre'] + \
            [f'mfcc_{i}' for i in range(len(features[0]) - 2)]
        df = pd.DataFrame(features, columns=feature_names)
        # find the date and time in format ddmmyyyy_hhmm
        date_time = datetime.datetime.now().strftime("%d%m%Y_%H%M")
        df.to_csv(f'features/mfcc/features_{date_time}.csv', index=False)
        print("Feature extraction complete. Saved to 'features/mfcc/features.csv'.")

    # Print list of files with errors
    if error_files:
        print(f"Errors occurred with the following files: {error_files}")
        # Optionally, write the error file paths to a text file for review
        with open('error_files.txt', 'w') as f:
            for file_path in error_files:
                f.write(f"{file_path}\n")
        print("List of problematic files saved to 'error_files.txt'.")


# Example usage
if __name__ == "__main__":
    base_path = "data/"
    process_all_audio_files(base_path)
