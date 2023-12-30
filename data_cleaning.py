import pandas as pd
import numpy as np
from tqdm import tqdm
from langdetect import detect, DetectorFactory

# Preprocessing to combine datasets
data_birth_year = pd.read_csv("data/birth_year.csv")
data_gender = pd.read_csv("data/gender.csv")

# The data is merged on the auhtor_ID column, after which all duplicate posts are dropped and the 'post_y' column is left out
merged_data = pd.merge(data_gender, data_birth_year, on='auhtor_ID')
removed_duplicates = merged_data.drop_duplicates(subset=['post_x'])

column_selection = removed_duplicates.drop(columns=['post_y'])
column_selection.reset_index(inplace=True, drop=True)
column_selection.rename(columns={"post_x": "post"}, inplace=True)

### ENGLISH FILTERING ###
def english_filtering():
    DetectorFactory.seed = 42
    post_list = list(column_selection["post"])
    language_list = []
    for post in tqdm(post_list):
        language_list.append(detect(post))

    column_selection["language"] = language_list
    column_selection.head()

    data_english = column_selection[column_selection["language"] == "en"]
    data_english.to_csv("data/combined_data_english.csv")

english_filtering()

### GENERATION FILTERING ###
combined = pd.read_csv("data\combined_data_english.csv")
generations = []
for i in range(len(combined)):
    if combined['birth_year'][i] in range(1980, 1997):
        generations.append(1)
    elif combined['birth_year'][i] in range(1997,2013):
        generations.append(0)
    else:
        generations.append(-1)

combined_gen = combined.copy()
combined_gen['Millennial'] = generations
combined_gen = combined_gen[combined_gen['Millennial']!=-1].reset_index(drop=True)

### BALANCING THE DATASET ###
majority_class = combined_gen[combined_gen['Millennial'] == 1]
minority_class = combined_gen[combined_gen['Millennial'] == 0]

# Randomly sample majority class to match minority class size
majority_downsampled = majority_class.sample(n=len(minority_class), random_state=42)
balanced_gen = pd.concat([majority_downsampled, minority_class])

print(balanced_gen['Millennial'].value_counts()) 
