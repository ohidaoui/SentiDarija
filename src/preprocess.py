import re
import pandas as pd
import string
from pyarabic.araby import strip_tatweel, strip_tashkeel



def identify_tweet_language(df, src_field='review'):
    """
    Takes a DataFrame with a 'tweet' column and adds a new 'language' column 
    that identifies whether each tweet is written in Arabic letters or Arabizi.
    """
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')  # pattern to match Arabic letters
    arabizi_pattern = re.compile(r'[a-zA-Z0-9]+')  # pattern to match Arabizi
    
    def identify_language(text):
        if arabic_pattern.search(text):
            return 'Arabic'
        elif arabizi_pattern.search(text):
            return 'Arabizi'
        else:
            return 'Unknown'
        
    new_df = df.copy()
    new_df['language'] = df[src_field].apply(identify_language)
    return new_df

def normalize_arabizi(text: str, normalize_dict: dict):
    words = text.split()
    for i, word in enumerate(words):
        for vocab_list in normalize_dict.values():
            for entry_list in vocab_list:
                if len(entry_list) > 1 and word in entry_list:
                    words[i] = entry_list[0]
    return ' '.join(words).strip()
    
def preprocess(src_df, source, field, d, arabizi=True):
    """
    Takes a DataFrame 'src_df' with a 'source' column and adds a new 'field' column 
    that represents a processed version of 'df[source]'
    """

    df = src_df.copy()
    df[field] = df[source].copy()
    df[field] = df[field].str.lower()
    
    df.drop_duplicates(subset=[field], inplace=True)
    # Replace URLs with URL string
    df[field] = df[field].replace(r'http\S+', 'URL', regex=True).replace(r'www\S+', 'URL', regex=True)
    # Replace user mentions with USER string
    df[field].replace(r'@[^\s]+', 'USER', regex=True, inplace=True)
    # Replace Hashtags with HASHTAG string
    df[field].replace(r'#[^\s]+', 'HASHTAG', regex=True, inplace=True)
    # Remove non-alphanumeric characters and digits
    df[field].replace(r'[^\w\s]', r'', regex=True, inplace=True)
    df[field].apply(lambda text: text.translate(str.maketrans("", "", string.punctuation)))
    df[field].replace(r'[^\w\s]|\d', r'', regex=True, inplace=True)
    # Replace with only one (remove repetitions)
    df[field].replace(r'(.)\1+', r'\1', regex=True, inplace=True)
    # Remove single letters 'h', 'w', ...
    df[field].replace(r'\b\w\b', r'', regex=True, inplace=True)
    df[field] = df[field].replace(r'  ', ' ', regex=True).apply(lambda s: s.strip())
    
    # Arabic specific processing
    # Remove Tatweel string
    df.loc[df.language=='Arabic', field] = df.loc[df.language=='Arabic', field].apply(strip_tatweel)
    # Remove Diacritics
    df.loc[df.language=='Arabic', field] = df.loc[df.language=='Arabic', field].apply(strip_tashkeel)
    
    # Arabizi specific processing
    # Normalization using DODa dataset : d is a dictionary extracted from DODa dataset
    if arabizi:
        df.loc[df.language=='Arabizi', field] = df.loc[df.language=='Arabizi', field].apply(normalize_arabizi, normalize_dict=d)
    
    return df
  
