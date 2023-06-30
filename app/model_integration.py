from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import re
import string
from pyarabic.araby import strip_tatweel, strip_tashkeel



def identify_reviews_language(df, src_field='review'):
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
    
def preprocess(src_df, source, field, d, arabizi=False):
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



# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT-mix")

def preprocessing_for_bert(data, max_len):

    encoded_sent = tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=data,  # Preprocess sentence
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=max_len,             # Max length to truncate/pad
                padding='max_length',           # Pad sentence to max length
                truncation=True,
                return_tensors='pt',            # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
    )
    
    return encoded_sent.get('input_ids').to('cpu'), encoded_sent.get('attention_mask').to('cpu')

def create_data_loader(reviews, max_len, batch_size):
    
    input_ids, masks = preprocessing_for_bert(reviews, max_len)
    dataset = TensorDataset(input_ids, masks)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    return dataloader

def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained DarijaBERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    model.to('cpu')

    all_logits = []
    
    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to('cpu') for t in batch)
        
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    
    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs


# Create the BertClassfier class
class SentiDarija(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    freeze_bert (bool): Set `False` to fine-tune the DarijaBERT model
        """
        super(SentiDarija, self).__init__()
        # Specify hidden size of DarijaBERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 64, 3

        # DarijaBERT
        self.bert = AutoModel.from_pretrained("SI2M-Lab/DarijaBERT-mix")

        # Instantiate an 2-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(H, D_out)
        )
        
        # Freeze DarijaBERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to DarijaBERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
    


# checkpoints = torch.load('model.pt')
checkpoints = torch.load('../model.pt')
loaded_model = SentiDarija()
loaded_model.load_state_dict(checkpoints['model_state_dict'])
loaded_model.to('cpu')