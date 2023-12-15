import pandas as pd
import re
from nltk import word_tokenize, pos_tag, pos_tag_sents
from collections import Counter

class FeatureExtractor: 

    def __init__(self, original_path, new_path):
        # Give path to the original data (do not change the original data)
        self.original_path = original_path

        # Give path to the new data
        self.new_path = new_path

        # Read in the original data
        self.original_df = pd.read_csv(original_path, index_col=[0])

    def add_feature(self, df_feature):
        # Read data + already existing features
        df_new = pd.read_csv(self.new_path, index_col=[0])

        # Join new feature on existing features
        df_new = df_new.join(df_feature)

        # Write new df with the feature to csv
        df_new.to_csv(self.new_path, index = True)
    
    def word_count(self):
        # Copy original df
        df_feature = self.original_df

        # Add new feature
        df_feature["word_count"] = df_feature["post"].apply(lambda n: len(n.split()))

        # Select only index and feature
        df_feature = df_feature[["word_count"]]

        # Add feature to csv
        self.add_feature(df_feature)

    def pos_count(self):
        noun_counts = []
        df_feature = self.original_df[:20]
        posts = df_feature['post'].tolist()
        tagged_posts = pos_tag_sents(map(word_tokenize, posts))
        df_feature['pos'] = tagged_posts
        
        for ind in df_feature.index:
            noun_count = 0
            for (word, tag) in df_feature['pos'][ind]:
                if tag.startswith('N'):
                    noun_count += 1
            noun_counts.append(noun_count)
        
        df_feature['noun_count'] = noun_counts
        print(df_feature)
    
    def drop_features(self, column_names: list): 
        df_features = pd.read_csv(self.new_path, index_col=[0])
        print(df_features)
        df_features = df_features.drop(columns= column_names)
        print(df_features)
        df_features.to_csv(self.new_path)


# Create an instance of the Feature Extractor with specified paths
FE = FeatureExtractor("data/cleaned_combined_data_english.csv", "data/final_combined_data_english.csv")

# Execute the function word_count and add the feature to the final dataset.
FE.pos_count()
#FE.drop_features(['word_count'])