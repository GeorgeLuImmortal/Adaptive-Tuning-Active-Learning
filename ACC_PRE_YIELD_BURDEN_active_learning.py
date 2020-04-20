import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE, ADASYN
import utilities as u
from numpy import genfromtxt
from optparse import OptionParser
from gensim.models import LdaModel
import smtplib
from email.mime.text import MIMEText


def main():

    parser = OptionParser(usage='usage: %prog [query methods] method1 method22... Choice are 0-random, 1-uncertainty, 2-certainty, 3-certaintyIG, 4-EGAL, 5-QBC, 6-DensityWeighted')

    
    parser.add_option("-p","--positive_dir", action="store", type="string", dest="dir_pos", help="directory of positive samples", default = 'animal_by_product_pos.csv')
    parser.add_option("-n","--negative_dir", action="store", type="string", dest="dir_neg", help="directory of negative samples", default = 'animal_by_product_neg.csv')
    parser.add_option("-m","--max_required", action="store", type="int", dest="required_max", help="the max number of required samples for labelling", default=120)
    parser.add_option("-i","--initial_samples", action="store", type="int", dest="samples_init", help="the number of initial samples of each class for training ",default=5)
    parser.add_option("-K","--labelled_samples", action="store", type="int", dest="samples_label", help="the number of samples moved from pool to training set each iteration",default=10)
    parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="output directory (default is result/ directory)", default='result/')
    parser.add_option("-r","--random_times", action="store", type="int", dest="randoms", help="the number of different independent runs", default=10)
    parser.add_option("-s","--start_random_state", action="store", type="int", dest="randoms_initial", help="the initial random state", default=1988)
    parser.add_option("-e","--num_of_estimators", action="store", type="int", dest="n_estimators", help="the number of estimators for QBC", default=5)
    parser.add_option("-g","--gridsearch_interval", action="store", type="int", dest="gridsearch_step", help="the number of iteration that requires a parameters tuning", default=10)
    parser.add_option("-t","--text_representation", action="store", type="choice", choices=('tfidf', 'tf', 'fasttext','fasttext_tuned','lda','bert'), 
        dest="text_rep", help="the text representations, options are: tfidf, tf, lda, fasttext, fasttext_tuned, bert (default is bert)", default='bert')



    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.error( "Must specify at least one selection method Choice are 0-random, 1-uncertainty, 2-certainty,3-certaintyIG, 4-EGAL, 5-QBC, 6-DensityWeighted" )

    for value in args:
        if value not in ['0','1','2','3','4','5','6']:
            parser.error( "wrong input Choice are 0-random, 1-uncertainty, 2-certainty, 3-certaintyIG, 4-EGAL, 5-QBC, 6-DensityWeighted" )


    (options, args) = parser.parse_args()


    max_queried = options.required_max

    models = [u.SvmModel, u.EnsembleModel]

    selections = [u.RandomSelection ,u.UncertaintySelection, u.CertaintySelection, u.CertiantyInformationGainSelection, u.EGAL, u.QBC, u.DensityWeighted] 

    selection_functions = [selections[int(i)] for i in args] 

    initial_samples = options.samples_init
    
    Ks = [options.samples_label] 


    ## for bert representations
    representations_neg = genfromtxt('./%s_data/'%(options.text_rep)+options.dir_neg, delimiter=',')
    representations_pos = genfromtxt('./%s_data/'%(options.text_rep)+options.dir_pos, delimiter=',')


    ulti_representations = np.concatenate((representations_neg,representations_pos),axis=0)
    labels = ['negative']*len(representations_neg)+['positive']*len(representations_pos)


    lda_neg = genfromtxt('./lda_data/'+options.dir_neg, delimiter=',')
    lda_pos = genfromtxt('./lda_data/'+options.dir_pos, delimiter=',')

    print('*'*100)
    print('*'*100)
    print('*'*100)
    print()
    print('Experiment setting information:')
    print()
    print('Postive', options.dir_pos, 'Negative', options.dir_neg)
    print('Text representation ', options.text_rep)
    print('Selection methods ', selection_functions)
    print('Max number of examples ', max_queried)
    print('Repeat times ', options.randoms)
    print('Initial random state', options.randoms_initial)
    print('Initial number of examples per class', options.samples_init)
    print('Number of examples moved from pool each time for labelling', options.samples_label)
    print('Number of iteration requiring a parameters tuning', options.gridsearch_step)
    print('Number of estimators used in QBC', options.n_estimators)
    print()
    doc_lda_matrix = np.concatenate((lda_neg,lda_pos),axis=0)
    print('lda shape',doc_lda_matrix.shape)
    print()
    le = preprocessing.LabelEncoder()
    class_ = list(set(labels))
    le.fit(class_)
    Y = le.transform(labels) 
    X = ulti_representations

    print('the shape of X is ', X.shape)
    print()
    print()
    print('*'*100)
    print('*'*100)
    print('*'*100)
   
    print()
    print()
    input("Press Enter to continue...")
    

    df_random_raw = pd.DataFrame()
    df_uncertainty_raw = pd.DataFrame()
    df_certainty_raw = pd.DataFrame()
    df_certaintyIG_raw = pd.DataFrame()
    df_EGAL_raw = pd.DataFrame()
    df_QBC_raw = pd.DataFrame()
    df_densityweighted_raw = pd.DataFrame()
   
    d = {}

    random_states=[i for i in range(options.randoms_initial,options.randoms_initial+options.randoms)]
    for idx,state in enumerate(random_states):
        
        
        print('-'*10,idx,'run of experiment','-'*10)
        X, Y = np.array(X), np.array(Y)

        print ('full pool:', X.shape, Y.shape)
        classes = len(np.unique(Y))
        print ('unique classes', classes)



        d = u.experiment(d, models, selection_functions, initial_samples,Ks ,max_queried,X, Y,state, options.gridsearch_step, doc_lda_matrix, options.n_estimators)


        if '0' in args:
            df_random_raw[idx]=d['SvmModel']['RandomSelection'][str(Ks[0])]['raw_result']
            df_random_raw.to_csv(options.dir_out+'raw_%s_%s_%s_random_gridsearch.csv'%(options.text_rep,options.dir_neg,options.dir_pos),header=[seed for seed in range(options.randoms_initial,options.randoms_initial+idx+1)])
           
        if '1' in args:
            df_uncertainty_raw[idx]=d['SvmModel']['UncertaintySelection'][str(Ks[0])]['raw_result']
            df_uncertainty_raw.to_csv(options.dir_out+'raw_%s_%s_%s_uncertainty_gridsearch.csv'%(options.text_rep,options.dir_neg,options.dir_pos),header=[seed for seed in range(options.randoms_initial,options.randoms_initial+idx+1)])
            

        if '2' in args:
            df_certainty_raw[idx]=d['SvmModel']['CertaintySelection'][str(Ks[0])]['raw_result']
            df_certainty_raw.to_csv(options.dir_out+'raw_%s_%s_%s_certainty_gridsearch.csv'%(options.text_rep,options.dir_neg,options.dir_pos),header=[seed for seed in range(options.randoms_initial,options.randoms_initial+idx+1)])
           

        if '3' in args:
            df_certaintyIG_raw[idx]=d['SvmModel']['CertiantyInformationGainSelection'][str(Ks[0])]['raw_result']
            df_certaintyIG_raw.to_csv(options.dir_out+'raw_%s_%s_%s_certaintyIG_gridsearch.csv'%(options.text_rep,options.dir_neg,options.dir_pos),header=[seed for seed in range(options.randoms_initial,options.randoms_initial+idx+1)])
            

        if '4' in args:
            df_EGAL_raw[idx]=d['SvmModel']['EGAL'][str(Ks[0])]['raw_result']
            df_EGAL_raw.to_csv(options.dir_out+'raw_%s_%s_%s_EGAL_gridsearch.csv'%(options.text_rep,options.dir_neg,options.dir_pos),header=[seed for seed in range(options.randoms_initial,options.randoms_initial+idx+1)])

        if '5' in args:
            df_QBC_raw[idx]=d['EnsembleModel']['QBC'][str(Ks[0])]['raw_result']
            df_QBC_raw.to_csv(options.dir_out+'raw_%s_%s_%s_QBC_gridsearch.csv'%(options.text_rep,options.dir_neg,options.dir_pos),header=[seed for seed in range(options.randoms_initial,options.randoms_initial+idx+1)])

        if '6' in args:
            df_densityweighted_raw[idx]=d['SvmModel']['DensityWeighted'][str(Ks[0])]['raw_result']
            df_densityweighted_raw.to_csv(options.dir_out+'raw_%s_%s_%s_densityweighted_gridsearch.csv'%(options.text_rep,options.dir_neg,options.dir_pos),header=[seed for seed in range(options.randoms_initial,options.randoms_initial+idx+1)])
            
            
        print('save to csv successfully for ',idx+1, ' run!!!!!')

       


      
    # mail_host = 'smtp.163.com'  
    # mail_user = '13631253525'  
    # mail_pass = '1122tdcQ.'   
    # sender = 'chuju03@163.com'  

    # receivers = ['jinghui.lu@ucdconnect.ie']  

    # message = MIMEText("task %s %s finish"%(options.text_rep,options.dir_neg[:-8]),'plain','utf-8')
     
    # message['Subject'] = "task %s %s finish"%(options.text_rep,options.dir_neg[:-8])

    # message['From'] = sender 
 
    # message['To'] = receivers[0]  

    # smtpObj = smtplib.SMTP() 
    # smtpObj.connect(mail_host,25)
    # smtpObj.login(mail_user,mail_pass) 
    # smtpObj.sendmail(sender,receivers,message.as_string()) 
    # smtpObj.quit() 
    # print('send success')   
        





    


    
   
   

# --------------------------------------------------------------

if __name__ == "__main__":
    main()

