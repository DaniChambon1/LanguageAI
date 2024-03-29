import pandas as pd
import re
from nltk import word_tokenize, pos_tag_sents
import string
import emoji
from data_cleaning import balanced_gen


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
    
    def drop_features(self, column_names: list): 
        df_features = pd.read_csv(self.new_path, index_col=[0])
        df_features = df_features.drop(columns= column_names)
        df_features.to_csv(self.new_path)
    
    def word_count(self):
        # Copy original df
        df_feature = self.original_df

        # Add new feature
        df_feature["word_count"] = df_feature["post"].apply(lambda n: len(word_tokenize(n)))

        # Select only index and feature
        df_feature = df_feature[["word_count"]]

        # Add feature to csv
        self.add_feature(df_feature)
    
    def contraction(self):
        # Copy original df and define contraction
        df_feature = self.original_df
        contraction = '[a-zA-Z]+\'m\s|[a-zA-Z]+\'d\s|[a-zA-Z]+\'ll\s|[a-zA-Z]+\'re\s|[a-zA-Z]+\'ve\s|[a-zA-Z]+\'s\s|[a-zA-Z]+\'t\s'
        lst_contractions = []

        # Loop over posts to find & count contractions
        for ind in df_feature.index:
            post = df_feature['post'][ind]
            find_contraction = re.findall(contraction, post)
            total_contractions_sentence = len(find_contraction) / len(word_tokenize(post))
            lst_contractions.append(total_contractions_sentence)
        
        # Add contractions to csv
        df_feature['contraction_count'] = lst_contractions
        df_feature = df_feature[['contraction_count']]
        self.add_feature(df_feature)

    def exaggeration(self):
        # Copy original df and define exaggeration        
        df_feature = self.original_df
        exaggeration = '[a|A]{3,}|[b|B]{3,}|[c|C]{3,}|[d|D]{3,}|[e|E]{3,}|[f|F]{3,}|[g|G]{3,}|[h|H]{3,}|[i|I]{3,}|[j|J]{3,}|[k|K]{3,}|[l|L]{3,}|[m|M]{3,}[n|N]{3,}|[o|O]{3,}|[p|P]{3,}|[q|Q]{3,}|[r|R]{3,}|[s|S]{3,}|[t|T]{3,}|[u|U]{3,}|[v|V]{3,}|[w|W]{3,}|[x|X]{3,}|[y|Y]{3,}|[z|Z]{3,}|[.]{2,}|[!]{2,}'
        lst_exaggeration = []

        # Loop over posts to find & count exaggerations
        for ind in df_feature.index:
            post = df_feature['post'][ind]
            find_exaggeration = re.findall(exaggeration, post)
            total_exaggeration_sentence = len(find_exaggeration) / len(word_tokenize(post))
            lst_exaggeration.append(total_exaggeration_sentence)

        # Add exaggerations to csv
        df_feature['exaggeration_count'] = lst_exaggeration
        df_feature = df_feature[['exaggeration_count']]
        self.add_feature(df_feature)
    
    def capital(self):
        # Copy original df
        df_feature = self.original_df
        percentage_capitals = []

        # Loop over posts to find & count capitals
        for ind in df_feature.index:
            count = 0
            for char in df_feature['post'][ind]:
                if char.isupper():
                    count += 1
            percentage_capitals.append((count * 100) / len(word_tokenize(df_feature['post'][ind])))
        
        # Add capitals to csv
        df_feature['percentage_capitals'] = percentage_capitals
        df_feature = df_feature[['percentage_capitals']]
        self.add_feature(df_feature)
    
    def emoticons(self):
        # Copy original df
        df_feature = self.original_df
        emoticon_count = []

        # Loop over posts to find & count emoticons
        for ind in df_feature.index:  
            emoticon_pattern = r'(:-?\)|:-?D|;-?\)|:-?P|:-?\(|:-?\/|:-?O|<3)'
            emoticons = re.findall(emoticon_pattern, df_feature['post'][ind])
            emoticon_count.append(len(emoticons) / len(word_tokenize(df_feature['post'][ind])))

        # Add emoticons to csv
        df_feature['emoticon_count'] = emoticon_count
        df_feature = df_feature[['emoticon_count']]
        self.add_feature(df_feature)
    
    def pronouns(self):
        # Copy original df 
        df_feature = self.original_df
        percentage_pronouns = []

        # Loop over posts to find & count pronouns
        for ind in df_feature.index:  
            pronoun_pattern = r'\b(?:he|she|it|they|we|you|I|me|my|mine|you|your|yours|him|her|hers|us|our|ours|them|their|theirs)\b'
            pronouns = re.findall(pronoun_pattern, df_feature['post'][ind])
            percentage_pronouns.append((len(pronouns) *100) / len(word_tokenize(df_feature["post"][ind])))
        
        # Add pronouns to csv
        df_feature['percentage_pronouns'] = percentage_pronouns
        df_feature = df_feature[['percentage_pronouns']]
        self.add_feature(df_feature)
    
    def punctuation(self):

        # Copy original df and define necessary objects
        df_feature = self.original_df
        punctuation_count = 0
        punctuation_counts = []
        comma_count = 0
        comma_counts = []
        exclamation_count = 0
        exclamation_counts = []

        # Loop over posts to find & count punctuation
        for post in list(df_feature["post"]):
            for character in post:
                if character in string.punctuation:
                    punctuation_count += 1
                if character == ",":
                    comma_count += 1
                if character == "!":
                    exclamation_count += 1
            
            punctuation_counts.append(punctuation_count / len(word_tokenize(post)))
            punctuation_count = 0

            comma_counts.append(comma_count / len(word_tokenize(post)))
            comma_count = 0

            exclamation_counts.append(exclamation_count / len(word_tokenize(post)))
            exclamation_count = 0

        # Add punctuation counts as columns
        df_feature['punctuation_counts'] = punctuation_counts
        df_feature['comma_counts'] = comma_counts
        df_feature['exclamation_counts'] = exclamation_counts

        df_feature = df_feature[['punctuation_counts', 'comma_counts', 'exclamation_counts']]
        self.add_feature(df_feature)

    def pos_count(self):
        # Declare lists for new columns
        noun_counts = []
        JJ_counts = []
        JJR_counts = []
        JJS_counts = []
        LS_counts = []
        MD_counts = []
        GM_counts = []
        RB_counts = []
        RBR_counts = []
        RBS_counts = []
        UH_counts = []
        VPR_counts = []
        VPA_counts = []

        # POS tagging
        df_feature = self.original_df
        posts = df_feature['post'].tolist()
        tagged_posts = pos_tag_sents(map(word_tokenize, posts))
        df_feature['pos'] = tagged_posts
        
        # Count POS
        for ind in df_feature.index:
            noun_count = 0 # Nouns
            JJ_count = 0 # Adjective
            JJR_count = 0 # Adjective, comparative
            JJS_count = 0 # Adjective, superlative
            LS_count = 0  # list item marker
            MD_count = 0 # Modal auxiliary
            GM_count = 0 # Genitive marker
            RB_count = 0 # Adverbs
            RBR_count = 0  # Adverbs, comparative
            RBS_count = 0 # Adverbs, superlative
            UH_count = 0 # Interjection
            VPR_count = 0 # Verbs, present
            VPA_count = 0 # Verbs, past

            # Count
            for (word, tag) in df_feature['pos'][ind]:
                if tag.startswith('N'):
                    noun_count += 1
                elif tag == ('JJ'):
                    JJ_count += 1
                elif tag == ('JJR'):
                    JJR_count += 1
                elif tag == ('JJS'):
                    JJS_count += 1
                elif tag == ('LS'):
                    LS_count += 1
                elif tag == ('MD'):
                    MD_count += 1
                elif tag == ('POS'):
                    GM_count += 1
                elif tag == ('RB'):
                    RB_count += 1
                elif tag == ('RBR'):
                    RBR_count += 1
                elif tag == ('RBS'):
                    RBS_count += 1
                elif tag == ('UH'):
                    UH_count += 1
                elif tag == ('VB') or tag == ('VBG') or tag == ('VBP') or tag == ('VBZ'):
                    VPR_count += 1
                elif tag == ('VBN') or tag == ('VBD'):
                    VPA_count += 1
            # Save counts
            noun_counts.append(noun_count/ len(word_tokenize(df_feature['post'][ind])))
            JJ_counts.append(JJ_count / len(word_tokenize(df_feature['post'][ind])))
            JJR_counts.append(JJR_count / len(word_tokenize(df_feature['post'][ind])))
            JJS_counts.append(JJS_count / len(word_tokenize(df_feature['post'][ind])))
            LS_counts.append(LS_count / len(word_tokenize(df_feature['post'][ind])))
            MD_counts.append(MD_count / len(word_tokenize(df_feature['post'][ind])))
            GM_counts.append(GM_count / len(word_tokenize(df_feature['post'][ind])))
            RB_counts.append(RB_count / len(word_tokenize(df_feature['post'][ind])))
            RBR_counts.append(RBR_count / len(word_tokenize(df_feature['post'][ind])))
            RBS_counts.append(RBS_count / len(word_tokenize(df_feature['post'][ind])))
            UH_counts.append(UH_count / len(word_tokenize(df_feature['post'][ind])))
            VPR_counts.append(VPR_count / len(word_tokenize(df_feature['post'][ind])))
            VPA_counts.append(VPA_count / len(word_tokenize(df_feature['post'][ind])))
        
        # Add counts to DF
        df_feature['noun_count'] = noun_counts
        df_feature['JJ_counts'] = JJ_counts 
        df_feature['JJR_counts'] = JJR_counts 
        df_feature['JJS_counts'] = JJS_counts
        df_feature['LS_counts'] = LS_counts
        df_feature['MD_counts'] = MD_counts
        df_feature['GM_counts'] = GM_counts
        df_feature['RB_counts'] = RB_counts
        df_feature['RBR_counts'] = RBR_counts
        df_feature['RBS_counts'] = RBS_counts
        df_feature['UH_counts'] = UH_counts 
        df_feature['VPR_counts'] = VPR_counts
        df_feature['VPA_counts'] = VPA_counts
        
        # Add counts to csv
        df_feature = df_feature[['noun_count','JJ_counts','JJR_counts','JJS_counts','LS_counts', 'MD_counts','GM_counts','RB_counts','RBR_counts', 'RBS_counts','UH_counts', 'VPR_counts', 'VPA_counts']]
        self.add_feature(df_feature)

    def emojis(self):
        # Copy original df
        df_feature = self.original_df
        emoji_count = []

        # Loop over posts to find & count emoticons
        for ind in df_feature.index:  
            emoji_count.append(emoji.emoji_count(df_feature['post'][ind]) / len(word_tokenize(df_feature['post'][ind])))

        # Add emoticons to csv
        df_feature['emoji_count'] = emoji_count
        df_feature = df_feature[['emoji_count']]
        self.add_feature(df_feature)



# initiate balanced_features.csv as balanced_gen.csv
balanced_gen = pd.read_csv("data/balanced_gen.csv", index_col=[0])
balanced_gen.to_csv("data/balanced_features.csv")
# Create an instance of the Feature Extractor with specified paths
FE = FeatureExtractor("data/balanced_gen.csv", "data/balanced_features.csv")
FE.word_count()
FE.contraction()
FE.exaggeration()
FE.capital()
FE.emoticons()
FE.pronouns()
FE.punctuation()
FE.pos_count()
FE.emojis()
print("Feature extraction done")
balanced_gen_features = pd.read_csv("data/balanced_features.csv", index_col=[0])