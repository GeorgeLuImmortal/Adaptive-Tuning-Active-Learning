import os
import re
import numpy as np
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm import tqdm
import nltk
from optparse import OptionParser
from gensim.models import fasttext
from gensim.models import Word2Vec
import gensim.downloader as api




def read_data(path):

    corpora = []
    for filename in os.listdir(path):

        df_temp = pd.read_csv(path+filename)

        corpora.append(df_temp.text.tolist())

    class_one_len = len(corpora[0])
    class_two_len = len(corpora[1])

    return corpora, class_one_len, class_two_len



def clean_corpus(corpus):
    
    cleaned_corpus = []
    
#     i = 0
    for article in corpus:
#         print(i)
        article = article.lower()
        temp_str = re.sub(r'\d+', '', article)
        temp_str = re.sub(r'[^\x00-\x7f]',r'', temp_str)
        temp_str = temp_str.translate(str.maketrans('', '', string.punctuation))
        temp_str = re.sub(r'\s+', ' ', temp_str)
        output =  re.sub(r"\b[a-zA-Z]{1,2}\b", "", temp_str)

        cleaned_corpus.append(output)
#         i+=1
        
    return cleaned_corpus

def generate_representation(corpora,path,model):
    
    all_corpus = corpora[0]+corpora[1]
    cleaned_corpus = all_corpus
#     cleaned_corpus = clean_corpus(all_corpus)

    print('total number of examples ',len(cleaned_corpus),'\n')
    
    representations=[]
    
    for article in cleaned_corpus:
        tokenized_article = article.split()
        corpus = [word for word in tokenized_article if word != '']
        matrix = []
    #     print(corpus)
        for word in corpus:
            try:
                matrix.append(model.wv[word])
            except Exception:
#                 print('out of vocabuary warning!!!')
                matrix.append(np.zeros((300)))
                pass
        matrix = np.array(matrix)
    #     print(matrix.shape)
        representations.append(np.mean(matrix,axis=0))
        
    representations = np.array(representations)

    print('represetation shape', representations.shape) 
    
    return representations


stemmer = PorterStemmer()
wnl = WordNetLemmatizer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (wnl.lemmatize(stemmer.stem(w)) for w in analyzer(doc))


def main():
    parser = OptionParser(usage='usage: %prog [encoding methods] method1 method2... Choice are 0-tf, 1-tfidf, 2-word2vec, 3-glove, 4-fasttext')
    parser.add_option("-i","--inputdir", action="store", type="string", dest="dir_input", help="Input directory of corpus for encoding (default is Statins)", default='Statins')
    
    

    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.error( "Must specify at least one selection method Choice are 0-tf, 1-tfidf, 2-word2vec, 3-glove, 4-fasttext" )

    for value in args:
        if value not in ['0','1','2','3','4']:
            parser.error( "wrong input Choice are 0-tf, 1-tfidf, 2-word2vec, 3-glove, 4-fasttext" )

    path = options.dir_input
    method = options
    corpora,class_one_len, class_two_len = read_data('./corpus_data/'+path+'/')


    # ## start preprocessing and encoding

    all_corpus = corpora[0]+corpora[1]

    cleaned_corpus = clean_corpus(all_corpus)
    print('total number of examples ',len(cleaned_corpus),'\n')

    print('-'*20,'the first example of class one before clean','-'*20)
    print(corpora[0][0])
    print('-'*20,'after clean','-'*20)
    print(cleaned_corpus[0])

    print('-'*20,'the first example of class two before clean','-'*20)
    print(corpora[1][0])
    print('-'*20,'after clean','-'*20)
    print(cleaned_corpus[class_one_len])

    if '1' in args:

        split_token = []
        for doc in cleaned_corpus:
            for word in doc.split():
                split_token.append(word)


        rare_terms = []
        freq_dist = nltk.FreqDist(split_token)
        for item in freq_dist.keys():
            if freq_dist[item]<10:
                rare_terms.append(item)
            
        
        final_corpus = []
        for i in tqdm(range(len(cleaned_corpus))):
            temp_doc = []
            for word in cleaned_corpus[i].split():
                if word not in rare_terms:
                    temp_doc.append(word)
                
            final_corpus.append(' '.join(temp_doc))




        count_vect = CountVectorizer(stop_words='english',binary=False,min_df=5,analyzer=stemmed_words)
        X_train_counts = count_vect.fit_transform(final_corpus)

        tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)


        representations = X_train_tf.toarray()

       
        print(path, 'TFIDF shape is (%s,%s)'%(X_train_tf.shape[0],X_train_tf.shape[1]))

        np.savetxt("./tfidf_data/%s_neg.csv"%(path), representations[:class_one_len], delimiter=",")
        np.savetxt("./tfidf_data/%s_pos.csv"%(path), representations[class_one_len:], delimiter=",")




    if '0' in args:

        split_token = []
        for doc in cleaned_corpus:
            for word in doc.split():
                split_token.append(word)


        rare_terms = []
        freq_dist = nltk.FreqDist(split_token)
        for item in freq_dist.keys():
            if freq_dist[item]<10:
                rare_terms.append(item)
            
        
        final_corpus = []
        for i in tqdm(range(len(cleaned_corpus))):
            temp_doc = []
            for word in cleaned_corpus[i].split():
                if word not in rare_terms:
                    temp_doc.append(word)
                
            final_corpus.append(' '.join(temp_doc))

        count_vect = CountVectorizer(stop_words='english',binary=False,min_df=5,analyzer=stemmed_words)
        X_train_counts = count_vect.fit_transform(final_corpus)

        matrix = X_train_counts.toarray()
        tf_representations = normalize(matrix, axis=1, norm='l1')

        
        print(path, 'TF shape is (%s,%s)'%(tf_representations.shape[0],tf_representations.shape[1]))


        np.savetxt("./tf_data/%s_neg.csv"%(path), tf_representations[:class_one_len], delimiter=",")
        np.savetxt("./tf_data/%s_pos.csv"%(path), tf_representations[class_one_len:], delimiter=",")

    if '2' in args:
        model = api.load('word2vec-google-news-300')
        print('---load model successfully---')

        representations = generate_representation(corpora,path,model)
    
        print(path, 'Word2vec shape is (%s,%s)'%(representations.shape[0],representations.shape[1]))
        np.savetxt("./w2v_data/%s_neg.csv"%(path), representations[:class_one_len], delimiter=",")
        np.savetxt("./w2v_data/%s_pos.csv"%(path), representations[class_one_len:], delimiter=",")

    if '3' in args:

        model = api.load('glove-wiki-gigaword-300')
        print('---load model successfully---')
        representations = generate_representation(corpora,path,model)
        print(path, 'Glove shape is (%s,%s)'%(representations.shape[0],representations.shape[1]))
        np.savetxt("./glove_data/%s_neg.csv"%(path), representations[:class_one_len], delimiter=",")
        np.savetxt("./glove_data/%s_pos.csv"%(path), representations[class_one_len:], delimiter=",")

    if '4' in args:

        model = fasttext.load_facebook_vectors('wiki.en.bin')
        print('---load model successfully---')
        
        representations = generate_representation(corpora,path,model)
        print(path, 'Fasttext shape is (%s,%s)'%(representations.shape[0],representations.shape[1]))
        np.savetxt("./fasttext_data/%s_neg.csv"%(path), representations[:class_one_len], delimiter=",")
        np.savetxt("./fasttext_data/%s_pos.csv"%(path), representations[class_one_len:], delimiter=",")



if __name__ == "__main__":
    main()

