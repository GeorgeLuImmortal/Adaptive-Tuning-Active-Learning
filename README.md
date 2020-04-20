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

The numbers specify the representation techniques used, 0-tf, 1-tfidf, 2-word2vec, 3-glove, 4-fasttext. -i specify the folder you save the raw input data.
It should be noted that you must specify at least one k, or you can input a list of k (*5 10 15 20 25 30*) as shown in above command. There are some other opitons you can specify such as output directory (default is directory "LDA_MODELS/"), random seed (default is 1984) and so on, as shown below
	
	python build_lda_model.py --random_state=2020 5 10 20 40 80 160 200
	
For more details, you can type command 	
	
	python build_lda_model.py -h

##### Step 2: Train Fasttext word embedding model

After generating LDA models, the next step of TBCC is to build up a word embedding model for LDA model selection procedure. 

	python build_fasttext_model.py
	
The word embedding model will be stored in directory "FASTTEXT_MODEL/" by default, you can also play with other parameters. By command 
*python build_fasttext_model.py -h* you can see other options such as window size, dimensions.

##### Step 3: Analysis of topic coherence of topic models

The third step is caculating the topic coherence of each topic model for selecting the model with highest topic coherece.
	
	python compute_semantic_coherence.py 5 10 20 40 80 160 200
	
The result will be stored in the directory "SEMANTIC_COH/". It should be noted the if you change the random seed when training LDA models (default random seed is 1984), you need to specify it explicitly, since the LDA models are named after random seed and topic numbers.

	python compute_semantic_coherence.py --random-state=2020 5 10 20 40 80 160 200

##### Step 4: Conduct topic-based corpus comparison

After choosing the best k, we can conduct a topic-based corpus comparison according to various statistical discrimination metrics.

	python topic_based_cc.py -k 200 -s 1984 -m jsd
	
It should be noted the parameters *k, s, m* is mandatory here indicating the number of topics, random_state (these two for targeting the topic model) and the employed statistical discrimination metrics (options are jsd, ext_jsd, chi, rf, ig, gr). The output will be stored in directory "COMPARISON_RESULT/" as well as shown in the console:


![alt text](https://github.com/GeorgeLuImmortal/topic-based_corpus_comparison/blob/master/COMPARISON_RESULT/comparison_result.png)


The first column is the index of topic, the second colmun is the words for charactering the topic, and the third colmun is the corpus index which the topic belongs to.

##### Step 5: Visualization

We can also visualizae the result via scatter plot exploiting t-SNE to project documents into a 2-D plane. The parameters are the same as above for example, k is the number of topic, s is the random state and m means the metric applied.

	python visualization.py -k 200 -s 1984 -m jsd

![alt text](https://github.com/GeorgeLuImmortal/topic-based_corpus_comparison/blob/master/VISUALIZATION/scatter_plot.png)

It will generate a html file under the directory "VISUALIZATION/" by default, you can open the file using any browser. The different color indicates documents from different corpora and the number is the index of the most discriminative topics selected by TBCC. You can refer to the file in directory "COMPARISON_RESULT/" for the contents of topics, or you can use mouth hover through the html page for the discriptors of the topics as shown in the above figure (this is a result of Chi-square, 200 topics).
