# imports
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
from tqdm import tqdm
from langdetect import detect


def english_filtering():
    combined_data = pd.read_csv("data/gender_age_combined.csv")

    # Create a list with the language of every post and add it as a column to the data

    post_list = list(combined_data["post"])
    language_list = []
    for post in tqdm(post_list):
        language_list.append(detect(post))

    combined_data["language"] = language_list
    combined_data.head()

    combined_data_english = combined_data[combined_data["language"] == "en"]
    combined_data_english.to_csv("data/combined_data_english.csv")

    combined_data.to_csv("data/combined_data_language.csv")

english_filtering()


