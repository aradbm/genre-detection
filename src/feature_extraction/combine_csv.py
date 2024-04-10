'''
Here we combine the csv files of the features extracted from the images.
We got:
    file_paths = ['features/chroma/features_29032024_1938.csv',
                  'features/mfcc/features_29032024_1930.csv']
We will combine these two csv files into one csv file.
'''

import pandas as pd

file_paths = ['features/chroma/features_29032024_1938.csv',
              'features/mfcc/features_29032024_1930.csv']

dfs = []

for file_path in file_paths:
    df = pd.read_csv(file_path)
    dfs.append(df)

# remove the 'genre' and 'file' columns from all but the first dataframe
for i in range(1, len(dfs)):
    dfs[i] = dfs[i].drop(columns=['genre', 'file'])

combined_df = pd.concat(dfs, axis=1)
combined_df.to_csv('features/combined_features.csv', index=False)
print("Combined features saved to features/combined_features.csv")
