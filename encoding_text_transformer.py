import torch
from transformers import *
import os
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
np.random.seed(2019)
import pandas as pd
from operator import add
import re
import string
from tqdm import tqdm_notebook, trange,tqdm
from optparse import OptionParser
import sys


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

def generate_representation(corpora, model_class, tokenizer_class, pretrained_weights):
    
   
    all_corpus = corpora[0]+corpora[1]
#     all_corpus = clean_corpus(all_corpus)
    print('total number of examples ',len(all_corpus),'\n')
    
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    model.cuda(1)
    
    representations = []
    representations_cls = []

    with torch.no_grad():
        iterator = tqdm(all_corpus, desc="Iteration")
        for idx,article in enumerate(iterator):

            tokenized_text = tokenizer.tokenize(article)


            if len(tokenized_text) > 511:
                split_index = len(tokenized_text)//511
#                 print(idx, split_index)
                temp_representations = []
                temp_representations_cls = []
                for i in range(split_index+1):
                    

                    temp_tokenized_text = ['[CLS]'] + tokenized_text[i*511:(i+1)*511]

                    indexed_tokens = tokenizer.convert_tokens_to_ids(temp_tokenized_text)
                    tokens_tensor = torch.tensor([indexed_tokens])
                    tokens_tensor = tokens_tensor.cuda(1)
                    encoded_layers = model(tokens_tensor)[0]
                    output_hidden = encoded_layers.cpu().numpy()


                    temp_representations.append(np.mean(output_hidden[0],axis=0))
                    temp_representations_cls.append(output_hidden[0][0])


                    del tokens_tensor,encoded_layers
                    torch.cuda.empty_cache()
                representations.append(temp_representations)
                representations_cls.append(temp_representations_cls)
            else:
#                 print(idx)
                
                tokenized_text = ['[CLS]'] + tokenized_text

                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                tokens_tensor = torch.tensor([indexed_tokens])
                tokens_tensor = tokens_tensor.cuda(1)
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

def main():
  

    parser = OptionParser(usage='usage: %prog [encoding methods] method1 method2... Choice are 0-bert, 1-gpt2 2-xlnet, 3-distilbert, 4-albert, 5-roberta')
    parser.add_option("-i","--inputdir", action="store", type="string", dest="dir_input", help="Input directory of corpus for encoding (default is Statins)", default='Statins')

    (options, args) = parser.parse_args()


    if len(args)<1:
        parser.error( "Must specify at least one selection method Choice are 0-bert, 1-gpt2 2-xlnet, 3-distilbert, 4-albert, 5-roberta" )

    for value in args:
        if value not in ['0','1','2','3','4','5']:
            parser.error( "wrong input Choice are 0-bert, 1-gpt2 2-xlnet, 3-distilbert, 4-albert, 5-roberta" )

    MODELS = [
              (0,BertModel,       BertTokenizer,       'bert-base-uncased'),
    # #           (BertModel,       BertTokenizer,       'bert-large-uncased'),
              (1,GPT2Model,       GPT2Tokenizer,       'gpt2'),
              (2,XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
              (3,DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
              (4,AlbertModel, AlbertTokenizer, 'albert-base-v2'),
              (5,RobertaModel,    RobertaTokenizer,    'roberta-base')]



    path = options.dir_input
     
    print('-'*10, path,'-'*10)

    corpora, class_one_len, class_two_len = read_data('./corpus_data/'+path+'/')

    print(len(corpora[0])+len(corpora[1]),' neg ', class_one_len, ' pos ', class_two_len)
    
    for idx, model_class, tokenizer_class, pretrained_weights in MODELS:

        if str(idx) in args:
            print('start encoding text by %s'%(pretrained_weights))

            representations, representations_cls = generate_representation(corpora,model_class, tokenizer_class, pretrained_weights)

            np.savetxt("./%s_data/%s_neg.csv"%(pretrained_weights,path), representations[:class_one_len], delimiter=",")
            np.savetxt("./%s_data/%s_pos.csv"%(pretrained_weights,path), representations[class_one_len:], delimiter=",")

            np.savetxt("./%s_data/%s_neg_cls.csv"%(pretrained_weights,path), representations_cls[:class_one_len], delimiter=",")
            np.savetxt("./%s_data/%s_pos_cls.csv"%(pretrained_weights,path), representations_cls[class_one_len:], delimiter=",")



if __name__ == "__main__":
    main()
