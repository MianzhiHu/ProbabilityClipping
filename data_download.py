import pandas as pd
import numpy as np
import os


# Preprocess the data
raw_data_list = []
for file in os.listdir('./delayed_discounting/raw_data/osfstorage/cleaned_behavioral_files'):
    if 'discount' in file:
        raw = pd.read_csv(f'./delayed_discounting/raw_data/osfstorage/cleaned_behavioral_files/{file}')
        raw_data_list.append(raw)
        print(f'{file} has {len(raw)} rows and {len(raw.columns)} columns.')

raw_data = pd.concat(raw_data_list, ignore_index=True)

# Remove NaN values in the 'choice' column
raw_data = raw_data.dropna(subset=['choice'])
print(raw_data.groupby('worker_id').size().describe())

# Remove false press
raw_data = raw_data[raw_data['choice'] != 'false']
print(raw_data.groupby('worker_id').size().describe())
print(np.sort(raw_data['large_amount'].unique()))
print(raw_data['large_amount'].nunique())
print(np.mean(raw_data['large_amount']))
print(raw_data['large_amount'].std())
print(np.sort(raw_data['small_amount'].unique()))
print(raw_data['small_amount'].nunique())
print(np.mean(raw_data['small_amount']))
print(raw_data['small_amount'].std())
print(np.sort(raw_data['later_delay'].unique()))
print(raw_data['later_delay'].nunique())
print(np.mean(raw_data['later_delay']))
print(raw_data['later_delay'].std())

summary_empirical = raw_data.groupby('worker_id')['choice'].value_counts(normalize=True).unstack()
raw_data.to_csv('./delayed_discounting/empirical_data/dd_empirical_data.csv', index=False)

