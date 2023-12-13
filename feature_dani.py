import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
from tqdm import tqdm
from langdetect import detect
import string

# This function does language detection using langdetect. It adds the column "language"
def english_filtering():
    combined_data = pd.read_csv("data/gender_age_combined.csv")
    language_data = combined_data.copy()
    # Create a list with the language of every post and add it as a column to the data
    post_list = list(language_data["post"])
    language_list = []
    for post in tqdm(post_list):
        language_list.append(detect(post))

    language_data["language"] = language_list
    language_data.head()

    # language_data_english = combined_data[combined_data["language"] == "en"]
    # language_data_english.to_csv("data/combined_data_english.csv")
    combined_data.to_csv("data/language_data.csv")

# english_filtering()

# This function does multiple kinds of punctuation detection
def punctuation():
    combined_data = pd.read_csv("data/gender_age_combined.csv")
    punctuation_data = combined_data.copy()
    post_list = list(punctuation_data["post"])
    
    punctuation_count_list = []
    punctuation_count = 0
    punctuation_count_standardized_list = []

    comma_count_list = []
    comma_count = 0
    comma_count_standardized_list = []

    exclamation_count_list = []
    exclamation_count = 0
    exclamation_count_standardized_list = []

    for post in tqdm(post_list):
        for character in post:
            if character in string.punctuation:
                punctuation_count += 1
            if character == ",":
                comma_count += 1
            if character == "!":
                exclamation_count += 1
        
        punctuation_count_list.append(punctuation_count)
        punctuation_count_standardized_list.append(punctuation_count / len(post))
        punctuation_count = 0

        comma_count_list.append(comma_count)
        comma_count_standardized_list.append(comma_count / len(post))
        comma_count = 0

        exclamation_count_list.append(exclamation_count)
        exclamation_count_standardized_list.append(exclamation_count /len(post))
        exclamation_count = 0

    punctuation_data["punctuation_count"] = punctuation_count_list
    punctuation_data["punctuation_count_standardized"] = punctuation_count_standardized_list
    
    punctuation_data["comma_count"] = comma_count_list
    punctuation_data["comma_count_standardized"] = comma_count_standardized_list
    
    punctuation_data["exclamation_count"] = exclamation_count_list
    punctuation_data["exclamation_count_standardized"] = exclamation_count_standardized_list
    
    punctuation_data.to_csv("data/punctuation_data.csv")

#punctuation()

punctuation_data = pd.read_csv("data/punctuation_data.csv")
print(punctuation_data.head())
print(punctuation_data["birth_year"].corr(punctuation_data["comma_count"]))

# commentje erbij