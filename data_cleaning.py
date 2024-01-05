import pandas as pd
from langdetect import detect, DetectorFactory

# Importing needed datasets
data_birth_year = pd.read_csv("data/birth_year.csv")
data_gender = pd.read_csv("data/gender.csv")

# The data is merged on the auhtor_ID column, after which all duplicate posts are dropped and the 'post_y' column is left out
merged_data = pd.merge(data_gender, data_birth_year, on='auhtor_ID')
removed_duplicates = merged_data.drop_duplicates(subset=['post_x'])

column_selection = removed_duplicates.drop(columns=['post_y']).rename(columns={"post_x": "post"})
column_selection.reset_index(inplace=True,drop=True)
## ENGLISH FILTERING ###
DetectorFactory.seed = 42
language_list = []
for post in list(column_selection["post"]):
    try:
        language_list.append(detect(post))
    except:
        language_list.append("unknown")

column_selection["language"] = language_list
combined_gen = column_selection[column_selection["language"] == "en"].copy()
combined_gen.reset_index(drop=True, inplace=True)

## GENERATION FILTERING ###
generations = []
for i in range(len(combined_gen)):
    if combined_gen['birth_year'][i] in range(1980, 1997):
        generations.append(1)
    elif combined_gen['birth_year'][i] in range(1997,2013):
        generations.append(0)
    else:
        generations.append(-1)  

combined_gen['Millennial'] = generations
combined_gen = combined_gen[combined_gen['Millennial']!=-1]
combined_gen.reset_index(inplace=True, drop=True)

### Defining minority and majority class
majority_class = combined_gen[combined_gen['Millennial'] == 1]
minority_class = combined_gen[combined_gen['Millennial'] == 0]

# Randomly sample majority class to match minority class size
majority_downsampled = majority_class.sample(n=len(minority_class), random_state=42)
balanced_gen = pd.concat([majority_downsampled, minority_class])
balanced_gen.reset_index(inplace=True, drop=True)
balanced_gen.to_csv("data/balanced_gen.csv")
print("Data cleaning done")
