import pandas as pd
import re

class FeatureExtractor: 

    def __init__(self, original_path, new_path):
        # Give path to the original data (do not change the original data)
        self.original_path = original_path

        # Give path to the new data
        self.new_path = new_path

        # Read in the original data
        self.original_df = pd.read_csv(original_path)

    def add_feature(self, df_feature):
        # Read data + already existing features
        df_new = pd.read_csv(self.new_path)

        # Join new feature on existing features
        df_new = df_new.join(df_feature, lsuffix = 'index', rsuffix='index')

        # Write new df with the feature to csv
        df_new.to_csv(self.new_path)
    
    def word_count(self):
        # Copy original df
        df_feature = self.original_df

        # Add new feature
        df_feature["word_count"] = df_feature["post"].apply(lambda n: len(n.split()))

        # Select only index and feature
        df_feature = df_feature[["index","word_count"]]

        # Add feature to csv
        self.add_feature(df_feature)

# Create an instance of the Feature Extractor with specified paths
FE = FeatureExtractor("data/cleaned_combined_data_english.csv", "data/final_combined_data_english.csv")

# Execute the function word_count and add the feature to the final dataset.
FE.word_count()