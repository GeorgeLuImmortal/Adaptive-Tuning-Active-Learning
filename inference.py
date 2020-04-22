
# coding: utf-8

# In[1]:


import torch
from transformers import *

import os

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd
from operator import add

import re
import string
from tqdm import tqdm_notebook, trange

# In[2]:


## remove digits, non-alphabetic chars, punctuations, word consists of less than 2 chars
## and extra spaces

def clean_corpus(corpus):
    
    cleaned_corpus = []
    
#     i = 0
    for article in corpus:
#         print(i)
#         article = article.lower()
        temp_str = re.sub(r'\d+', '', article)
        temp_str = re.sub(r'[^\x00-\x7f]',r'', temp_str)
        temp_str = temp_str.translate(str.maketrans('', '', string.punctuation))
        temp_str = re.sub(r'\s+', ' ', temp_str)
#         output =  re.sub(r"\b[a-zA-Z]{1,2}\b", "", temp_str)

        cleaned_corpus.append(temp_str)
#         i+=1
        
    return cleaned_corpus

def read_data(path):

    corpora = []
    for filename in os.listdir(path):

        df_temp = pd.read_csv(path+filename)

        corpora.append(df_temp.text.tolist())

    class_one_len = len(corpora[0])
    class_two_len = len(corpora[1])

    return corpora, class_one_len, class_two_len

def generate_representation(corpora, model_class, tokenizer_class, pretrained_weights, checkpoint):
    
   
    all_corpus = corpora[0]+corpora[1]
#     all_corpus = clean_corpus(all_corpus)
    print('total number of examples ',len(all_corpus),'\n')
    
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(checkpoint)
    model.to('cuda')
    
    representations = []
    representations_cls = []

    with torch.no_grad():
        iterator = tqdm_notebook(all_corpus, desc="Iteration")
        for idx,article in enumerate(iterator):

            tokenized_text = tokenizer.tokenize(article)


            if len(tokenized_text) > 511:
                split_index = len(tokenized_text)//511
                # print(idx, split_index)
                temp_representations = []
                temp_representations_cls = []
                for i in range(split_index+1):
                    

                    temp_tokenized_text = ['[CLS]'] + tokenized_text[i*511:(i+1)*511]

                    indexed_tokens = tokenizer.convert_tokens_to_ids(temp_tokenized_text)
                    tokens_tensor = torch.tensor([indexed_tokens])
                    tokens_tensor = tokens_tensor.to('cuda') 
                    encoded_layers = model(tokens_tensor)[0]
                    output_hidden = encoded_layers.cpu().numpy()


                    temp_representations.append(np.mean(output_hidden[0],axis=0))
                    temp_representations_cls.append(output_hidden[0][0])


                    del tokens_tensor,encoded_layers
                    torch.cuda.empty_cache()
                representations.append(temp_representations)
                representations_cls.append(temp_representations_cls)
            else:
                # print(idx)
                
                tokenized_text = ['[CLS]'] + tokenized_text

                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                tokens_tensor = torch.tensor([indexed_tokens])
                tokens_tensor = tokens_tensor.to('cuda')
                encoded_layers = model(tokens_tensor)[0]
                encoded_layers = encoded_layers.cpu().numpy()

                representations.append(np.mean(encoded_layers[0],axis=0))
                representations_cls.append(encoded_layers[0][0])

                del tokens_tensor,encoded_layers
                torch.cuda.empty_cache()


    ulti_representations = []
    for representation in representations:
        if type(representation)==list:
            ulti_representations.append(np.mean(representation,axis=0))
        else:
            ulti_representations.append(representation)
            
            
    ulti_representations_cls = []
    for representation in representations_cls:
        if type(representation)==list:
            ulti_representations_cls.append(np.mean(representation,axis=0))
        else:
            ulti_representations_cls.append(representation)

    return ulti_representations, ulti_representations_cls




# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [
#           (BertModel,       BertTokenizer,       'bert-base-uncased'),
#           (BertModel,       BertTokenizer,       'bert-large-uncased'),
#           (GPT2Model,       GPT2Tokenizer,       'gpt2'),
#           (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
#           (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
#           (AlbertModel, AlbertTokenizer, 'albert-base-v2'),
          (RobertaModel,    RobertaTokenizer,    'roberta-base')]



def inference(dataset_name,checkpoint,loop):



 
    print('-'*10, dataset_name,'-'*10)

    corpora, class_one_len, class_two_len = read_data('./corpus_data/'+dataset_name+'/')

    print(len(corpora[0])+len(corpora[1]),' neg ', class_one_len, ' pos ', class_two_len)
    
    for model_class, tokenizer_class, pretrained_weights in MODELS:
        print('start encoding text by %s'%(pretrained_weights))

        representations, representations_cls = generate_representation(corpora,model_class, tokenizer_class, pretrained_weights,checkpoint)

        np.savetxt("./%s_data/%s_tuned_neg_%s.csv"%(pretrained_weights,dataset_name,loop), representations[:class_one_len], delimiter=",")
        np.savetxt("./%s_data/%s_tuned_pos_%s.csv"%(pretrained_weights,dataset_name,loop), representations[class_one_len:], delimiter=",")

        np.savetxt("./%s_data/%s_tuned_neg_cls_%s.csv"%(pretrained_weights,dataset_name,loop), representations_cls[:class_one_len], delimiter=",")
        np.savetxt("./%s_data/%s_tuned_pos_cls_%s.csv"%(pretrained_weights,dataset_name,loop), representations_cls[class_one_len:], delimiter=",")




