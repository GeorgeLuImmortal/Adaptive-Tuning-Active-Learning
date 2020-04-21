# Effectiveness-of-Pretrained-Transformer-based-Language-Models-in-Active-Learning-for-Labelling-Data
Investigating the Effectiveness of Representations Based on Pretrained Transformer-based Language Models in Active Learning for Labelling Text Datasets


### Dependencies
Tested Python 3.6, and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.16.4](http://www.numpy.org/)
* Required: [scikit-learn >= 0.21.1](http://scikit-learn.org/stable/)
* Required: [pandas >= 0.25.1](https://pandas.pydata.org/)
* Required: [gensim >= 3.7.3](https://radimrehurek.com/gensim/)
* Required: [matplotlib >= 2.2.2](https://matplotlib.org/)
* Required: [torch >= 1.3.1](https://pytorch.org/)
* Required: [transformers >= 2.2.2](https://huggingface.co/transformers/)
* Required: [FastText model trained with Wikipedia 300-dimension](https://fasttext.cc/docs/en/pretrained-vectors.html)


### Basic Usage

To perform active learning for text labellling, the input corpus of documents should consist of plain text files stored in csv format (two files for one corpus, one for documents belong to class A and one for documents for class B), each row corresponding to one document in that corpus, the format can be refered to the csv file in the sample directory "corpus_data/ProtonPumpInhibitors/".

##### Step 1: Encoding Text

The first step of the system is converting raw text data into vectorized format, the raw text data located in directory "corpus_data/", each dataset should have its individual directory, for example, the "ProtonPumpInhibitors" under folder "corpus_data".  The input corpus of documents should consist of plain text files stored in csv format (two files for one corpus, one for documents belong to class A and one for documents for class B), each row corresponding to one document in that corpus, the format can be refered to the csv file in the sample directory "corpus_data/ProtonPumpInhibitors/". Then we can start convert the text into vectors:

	python encoding_text.py 0 1 2 3 4 -i ProtonPumpInhibitors

The numbers specify the representation techniques used, 0-tf, 1-tfidf, 2-word2vec, 3-glove, 4-fasttext and -i specify the folder you save the raw input data.
If you want to convert text into embeddings by transformer-based model:

	python encoding_text_transformer.py 0 1 2 3 4 5 -i ProtonPumpInhibitors
	
The numbers specify the representation techniques used, 0-bert, 1-gpt2, 2-xlnet, 3-distilbert, 4-albert, 5-roberta and -i specify the folder you save the raw input data. It should be noted, for each corpus, this script will generate two sort of representation, one is averaged embeddings, another is "[CLS]" token embeddings which is suffixed by "cls.csv".

##### Step 2: Active Learning

We can start active learning procedure by:

	ACC_PRE_YIELD_BURDEN_active_learning.py 0 1 2 3 4 5 -t roberta-base -m 1000 -r 10 -n ProtonPumpInhibitors_neg.csv -p ProtonPumpInhibitors_pos.csv

The number indicates the selection methods 0-random, 1-uncertainty, 2-certainty, 3-certainty-informationGain, 4-EGAL, 5-QBC, 6-InformationDensity, -t specify the text representation, options are w2v, glove, fasttext, tf, tfidf, bert, roberta-base, disilbert-base-uncased, gpt2, xlnet-base-cased, albert-base-v2. 
-m specify the max number of instances labelled, -r means the number of repetition with different random seed, -n, -p indicates the encoded input from different classes. Other arguments like number of documents labelled per iteration, number of estimators for QBC can be found by --help or -h.

##### Step 3: Experimental Results

We investigate the impact of different representations in active learning, we compare some popular pretrained transformer-based models such as Bert, Roberta with some classicial techniques such as tfidf, word2vec. We apply different selection methods in active learning to mitigate the potential impacts caused by selection method. The following picture visualize the performance of different representations in Multi-Domain Customer Review dataset, 

![alt text](https://github.com/GeorgeLuImmortal/Effectiveness-of-Pretrained-Transformer-based-Language-Models-in-Active-Learning-for-Labelling-Data/blob/master/plots/pretrained_Longer_MultidomainCustomerReview_uncertainty.png)
