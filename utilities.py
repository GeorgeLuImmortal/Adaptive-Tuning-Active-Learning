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
from numpy import genfromtxt
from optparse import OptionParser
from sklearn.model_selection import ShuffleSplit,StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import operator
from sklearn.ensemble import BaggingClassifier




def compute_p_nd(lda_labelled,lda_val):
    
        p_nds = {}
        
        S = lda_labelled
        
        pca = PCA(n_components=3,random_state=2019)
        pca.fit(S)
        
        U = pca.components_
        UU = np.matmul(U.transpose(),U)
        
        for idx,doc in enumerate(lda_val):
            numerator = np.linalg.norm(np.matmul(UU,doc))
            denominator = np.linalg.norm(doc)
            p_nd = 1-numerator/denominator
            p_nds[idx] = p_nd
            
        return p_nds

def compute_density(similarities,alpha):
    density_matrix = similarities.copy()
    density_matrix[density_matrix<alpha]=0
    density = np.sum(density_matrix, axis=1)
    
    return density

def compute_candidate_set(unlabelled_samples,labelled_samples,w):
    u_l_sim = cosine_similarity(unlabelled_samples,labelled_samples)
    max_sim = np.max(u_l_sim,axis=1)
    sort_max_sim = np.sort(max_sim)
   
    loc = int(w*len(sort_max_sim))
    beta = sort_max_sim[loc]
    
    
    permutation = np.where(max_sim<beta)[0]
    
    return permutation


def compute_vote_entropy(results):
    
    vote_entropies = []
    
    n_estimators = results.shape[1]
    n_examples = results.shape[0]
    
    v_pos = np.sum(results, axis = 1)/n_estimators
    v_neg = 1-v_pos
    
    v_pos = v_pos*np.nan_to_num(np.log2(v_pos))
    v_neg = v_neg*np.nan_to_num(np.log2(v_neg))
    
    denominator = np.log2(min(n_estimators, 2))
    
    ve = -1/denominator*(v_pos+v_neg)
    
    return v_pos, v_neg, ve

class BaseModel(object):

    def __init__(self):
        pass

    def fit_predict(self):
        pass


class SvmModel(BaseModel):

    model_type = 'Support Vector Machine'
    def fit_predict(self, X_train, y_train, X_val, c_weight ,active_iteration, gridsearch_interval,random_state, n_estimators):
        print ('training svm ...')
        if active_iteration%gridsearch_interval==0:
            print('start gridsearch ...')
            parameters = [
                         #    {'kernel': ['rbf'], 'gamma': ['scale'],
                         # 'C': [ 0.01, 0.1, 1, 10, 100]},
                        # {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                        #  'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                        {'kernel': ['linear'],
                         'C': [ 0.01, 0.1, 1,10]}
                       ]
    #         parameters = {'kernel':('linear', 'rbf','poly'), 'C':Cs, 'gamma':gammas}
    #         self.classifier = SVC(C=1, kernel='linear', probability=True,
    #                               class_weight=c_weight,random_state=2019)
            cv = StratifiedKFold(n_splits=5,random_state=random_state)
            svc = SVC(probability=True,random_state=2019,class_weight=c_weight,max_iter=10000)
            self.classifier = GridSearchCV(svc, parameters, cv=cv,scoring='accuracy',n_jobs=8,verbose = 1)
            self.classifier.fit(X_train, y_train)
            print('best parameters is ', self.classifier.best_params_)
            self.best_params_ = self.classifier.best_params_
            self.val_y_predicted = self.classifier.predict(X_val)
            return (X_train, X_val, self.val_y_predicted)
        else:
            kernel = self.best_params_['kernel']
            C = self.best_params_['C']
            # gamma = self.best_params_['gamma']

            # self.classifier = SVC(probability=True,random_state=2019,class_weight=c_weight,C=C,kernel=kernel,gamma=gamma)  
            self.classifier = SVC(probability=True,random_state=2019,class_weight=c_weight,C=C,kernel=kernel,max_iter=10000)
            self.classifier.fit(X_train, y_train)
            print('best parameters is ', self.classifier)
            self.val_y_predicted = self.classifier.predict(X_val)
            return (X_train, X_val, self.val_y_predicted)


class EnsembleModel(BaseModel):

    model_type = 'Ensemble Model'
    def fit_predict(self, X_train, y_train, X_val, c_weight ,active_iteration, gridsearch_interval,random_state, n_estimators):
        print ('training bagging classifiers ...')

        self.classifier = BaggingClassifier(SVC(probability=True,random_state=2019,class_weight=c_weight,C=5,kernel='linear',max_iter=10000),
                             max_samples=1.0, max_features=1.0, random_state=random_state, n_estimators = n_estimators, n_jobs=8)

        self.classifier.fit(X_train, y_train)
        self.val_y_predicted = self.classifier.predict(X_val)

        
        return (X_train, X_val, self.val_y_predicted)
    
            






class TrainModel:

    def __init__(self, model_object):        
        self.accuracies = []
        self.precisions = []
        self.burdens = []
        self.yields = []
        self.raw_result = []
        self.model_object = model_object()        

    def print_model_type(self):
        print (self.model_object.model_type)

    # we train normally and get probabilities for the validation set. i.e., we use the probabilities to select the most uncertain samples

    def train(self, X_train, y_train, X_val, c_weight, active_iteration, gridsearch_interval, random_seed, n_estimators):
        print ('Train set:', X_train.shape, 'y:', y_train.shape)
        print ('Val   set:', X_val.shape)
     
        t0 = time.time()
        (X_train, X_val, self.val_y_predicted) =  self.model_object.fit_predict(X_train, y_train, X_val, c_weight, active_iteration, gridsearch_interval,random_seed, n_estimators)
        self.run_time = time.time() - t0
        return (X_train, X_val)  # we return them in case we use PCA, with all the other algorithms, this is not needed.


    # we want accuracy only for the whole dataset
    def get_accuracy(self, y_val,TP_h,TN_h,N,iteration):
#         classif_rate = np.mean(self.test_y_predicted.ravel() == y_test.ravel()) * 100
#         classif_rate = precision_score(y_test, self.test_y_predicted,pos_label=1)
        tn, fp, fn, tp = confusion_matrix(y_val, self.val_y_predicted,labels=[0,1]).ravel()
        accu_plus = (tp+TP_h+tn+TN_h)/N
        precision_plus = (tp+TP_h)/(tp+TP_h+fp)
        burden = (TP_h+TN_h+tp+fp)/N
        yield_ = (TP_h + tp)/(TP_h+tp+fn)

        result_dist = {'TP_H':TP_h,'TN_H':TN_h,'TP_M':tp, 'TN_M':tn,'FP_M':fp,'FN_M':fn}
        self.raw_result.append(result_dist) 

#         print(classif_rate)
        self.accuracies.append(accu_plus)   
        self.precisions.append(precision_plus)
        self.burdens.append(burden)
        self.yields.append(yield_)
        
        print('--------------------------------')
        print('Activation Iteration:',iteration)
        print('Assigned label result TP_human:',TP_h,'TN_human', TN_h)
        print('Predict result TN:',tn,'FP:', fp,'FN:', fn,'TP:', tp)
        print('accuracy_plus is %.3f' % accu_plus,'\n')
        print('precision_plus is %.3f ' % precision_plus,'\n')
        print('yield is %.3f ' % yield_,'\n')
        print('burden is %.3f ' % burden,'\n')
        
        print('--------------------------------')
        print('y-val set:',y_val.shape)
        print('Example run in %.3f s' % self.run_time,'\n')
#         print("Precision rate for %.3f " % (classif_rate))    
        print("Classification report for classifier %s:\n%s\n" % (self.model_object.classifier, metrics.classification_report(y_val, self.val_y_predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_val, self.val_y_predicted,labels=[0,1]))
        print('--------------------------------')
        





class BaseSelectionFunction(object):

    def __init__(self):
        pass

    def select(self):
        pass


class RandomSelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, k,labelled_index, lda_val,unlabelled_samples,labelled_samples,
           density, dist, vote_entropies):
        random_state = 1024
        np.random.seed(random_state)
        selection = np.random.choice(probas_val.shape[0], k, replace=False)

#     print('uniques chosen:',np.unique(selection).shape[0],'<= should be equal to:',initial_labeled_samples)

        return selection


class UncertaintySelection(BaseSelectionFunction):

    @staticmethod
    # def select(probas_val, k,lda_labelled, lda_val,unlabelled_samples,labelled_samples,
    #        density,dist):
    #     e = (-probas_val * np.log2(probas_val)).sum(axis=1)
    #     selection = (np.argsort(e)[::-1])[:k]
    #     return selection
    def select(probas_val, k, lda_labelled, lda_val,unlabelled_samples,labelled_samples,
        density, dist, vote_entropies):
        selection = np.argsort(dist)[:k]

        return selection

class CertaintySelection(BaseSelectionFunction):

    @staticmethod
    # def select(probas_val, k,lda_labelled, lda_val,unlabelled_samples,labelled_samples,
    #        density,dist):
    #     e = (-probas_val * np.log2(probas_val)).sum(axis=1)
    #     selection = (np.argsort(e))[:k]
    #     return selection
    def select(probas_val, k, lda_labelled, lda_val,unlabelled_samples,labelled_samples,
    density, dist, vote_entropies):
        selection = np.argsort(dist)[::-1][:k]

        return selection
      
class CertiantyInformationGainSelection(BaseSelectionFunction):


    @staticmethod
    def select(probas_val, k, lda_labelled, lda_val,unlabelled_samples,labelled_samples,
           density, dist, vote_entropies):
        p = {}
        
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        p_nds = compute_p_nd(lda_labelled,lda_val)
        
        for key in p_nds.keys():
            p[key] = p_nds[key]*e[key]
            
        sorted_score = np.array(sorted(p.items(), key=operator.itemgetter(1)))
        selection = sorted_score[-k:,0]
        selection = [int(value) for value in selection]
        return selection


class EGAL(BaseSelectionFunction):


    @staticmethod
    def select(probas_val, k, lda_labelled, lda_val,unlabelled_samples,labelled_samples,
        density, dist, vote_entropies):
        w = 0.25
    
        candidate_index = compute_candidate_set(unlabelled_samples,labelled_samples,w)
    
        temp_density = density[candidate_index]
    
        density_index = np.argsort(temp_density)[-k:]
    
        selection = candidate_index[density_index]

        return selection


class QBC(BaseSelectionFunction):


    @staticmethod
    def select(probas_val, k, lda_labelled, lda_val,unlabelled_samples,labelled_samples,
        density, dist, vote_entropies):
        w = 0.25
    
        selection = np.argsort(vote_entropies)[::-1][:k]

        return selection

class DensityWeighted(BaseSelectionFunction):


    @staticmethod
    def select(probas_val, k, lda_labelled, lda_val,unlabelled_samples,labelled_samples,
        density, dist, vote_entropies):
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        similarities = cosine_similarity(unlabelled_samples)
        avg_sim = np.mean(similarities, axis=1)

        density_weighted = e*avg_sim

        selection = np.argsort(density_weighted)[::-1][:k]



        return selection


class Normalize(object):
    
    def normalize(self, X_train, X_val):
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
       
        return (X_train, X_val) 
    
    def inverse(self, X_train, X_val):
        X_train = self.scaler.inverse_transform(X_train)
        X_val   = self.scaler.inverse_transform(X_val)
    
        return (X_train, X_val) 





def get_k_random_samples(initial_samples, X,
                         Y,random_state):
    
    np.random.seed(random_state)

    df = pd.DataFrame(X)
    df['label'] = Y

    Samplesize = initial_samples  #number of samples that you want       
    initial_samples = df.groupby('label', as_index=False).apply(lambda array: array.loc[np.random.choice(array.index, Samplesize, False),:])

    permutation = initial_samples.index.levels[1]
    
    
   
    print ('initial random chosen samples', permutation),
#            permutation)
    X_train = X[permutation]
    y_train = Y[permutation]
    X_train = X_train.reshape((X_train.shape[0], -1))
    bin_count = np.bincount(y_train.astype('int64'))
    unique = np.unique(y_train.astype('int64'))
    print (
        'initial train set:',
        X_train.shape,
        y_train.shape,
        'unique(labels):',
        bin_count,
        unique,
        )
    tp_h = bin_count[1]
    tn_h = bin_count[0]
    return (permutation, X_train, y_train, tp_h, tp_h)





class TheAlgorithm(object):

    

    def __init__(self, initial_samples,k,model_object, selection_function,max_queried,gridsearch_interval,doc_lda_matrix, n_estimators):
        self.initial_samples = initial_samples
        self.k = k
        self.model_object = model_object
        self.sample_selection_function = selection_function
        self.max_queried = max_queried
        self.gridsearch_interval = gridsearch_interval
        self.doc_lda_matrix = doc_lda_matrix
        self.n_estimators = n_estimators

    def run(self, X, Y,random_seed):

       
        
        active_iteration = 0
        print('-'*5,'start training: random_seed:',random_seed,'selection method:',self.sample_selection_function.__name__)
        N = len(X)
        TP_h = 0
        TN_h = 0
        # initialize process by applying base learner to labeled training data set to obtain Classifier
        
        ## initial_labeled_samples: the number of samples at the begining
        (permutation, X_train, y_train,tp_h,tn_h) =             get_k_random_samples(self.initial_samples,
                                 X, Y,random_seed)

        
        self.queried = self.initial_samples*2
        self.samplecount = [self.k]
        TP_h = TP_h + tp_h
        TN_h = TN_h + tn_h


        ## compute similarities, sigma, mu, alpha and density for EGAL
        similarities = cosine_similarity(X)

        sigma = np.std(similarities)
        mu = np.mean(similarities)
        alpha = mu-0.5*sigma

        density = compute_density(similarities,alpha)

        

        # assign the val set the rest of the 'unlabelled' training data

        X_val = np.array([])
        y_val = np.array([])
        X_val = np.copy(X)
        X_val = np.delete(X_val, permutation, axis=0)
        y_val = np.copy(Y)
        y_val = np.delete(y_val, permutation, axis=0)
        print ('val set:', X_val.shape, y_val.shape, permutation.shape)
        print ()
 
        # assign the val set of lda the rest of the 'unlabelled' training data

        lda_val = np.array([])
        lda_val = np.copy(self.doc_lda_matrix)
        lda_val = np.delete(lda_val, permutation, axis=0)

        lda_labelled = self.doc_lda_matrix[permutation]
        
        print ('lda val set:', lda_val.shape, permutation.shape)
        print ('lda labelled set:', lda_labelled.shape, permutation.shape)
        print ()

        # assign the val set of density the rest of the 'unlabelled' training data

        density_val = np.array([])
        density_val = np.copy(density)
        density_val = np.delete(density_val, permutation, axis=0)
        
        print ('density val set:', density_val.shape, permutation.shape)
        print ()

        ## upsampling the data
        
#         ros = BorderlineSMOTE(random_state=0)
#         X_train_resampled, y_train = ros.fit_resample(X_train, y_train)

        # normalize data

        normalizer = Normalize()
        X_train, X_val = normalizer.normalize(X_train, X_val)   
        
        self.clf_model = TrainModel(self.model_object)
        (X_train, X_val) = self.clf_model.train(X_train, y_train, X_val, 'balanced',active_iteration,self.gridsearch_interval, random_seed, self.n_estimators)
     
        self.clf_model.get_accuracy(y_val,TP_h,TN_h,N,active_iteration)

        # fpfn = self.clf_model.test_y_predicted.ravel() != y_val.ravel()
        # print(fpfn)
        # self.fpfncount = []
        # self.fpfncount.append(fpfn.sum() / y_test.shape[0] * 100)

        while self.queried < self.max_queried:

            active_iteration += 1

            # get validation probabilities

            probas_val = self.clf_model.model_object.classifier.predict_proba(X_val)
            # predict_val = self.clf_model.model_object.classifier.predict(X_val)
            print ('val predicted:',
                   self.clf_model.val_y_predicted.shape,
                   self.clf_model.val_y_predicted)

            # print ('val predicted:',
            #        predict_val.shape,
            #        predict_val)

            print ('probabilities:', probas_val.shape, '\n',
                    # probas_val,
                   np.argmax(probas_val, axis=1))


            if self.sample_selection_function.__name__ == 'QBC':

                # get the Vote Entropy of validation set
                predict_results = np.zeros((X_val.shape[0],self.n_estimators) )
                for idx, estimators in enumerate(self.clf_model.model_object.classifier.estimators_) :
                    predict_results[:,idx] = estimators.predict(X_val)

                _, _, vote_entropies = compute_vote_entropy(predict_results)


                # do not neet to compute distance
                dist = 0

            else:

                # get the distance from hyperplane
                numerators = self.clf_model.model_object.classifier.decision_function(X_val)
                try:
                    w_norm = np.linalg.norm(self.clf_model.model_object.classifier.best_estimator_.coef_)
                except Exception:
                    w_norm = np.linalg.norm(self.clf_model.model_object.classifier.coef_)

                dist = abs(numerators) / w_norm

                # do not neet to compute vote_entropies
                vote_entropies = 0

            # select samples using a selection function
            # normalization needs to be inversed and recalculated based on the new train and test set.
            X_train, X_val = normalizer.inverse(X_train, X_val)  

            # get the uncertain samples from the validation set
            uncertain_samples =  self.sample_selection_function.select(probas_val, self.k, lda_labelled, lda_val, X_val, X_train, density ,dist, vote_entropies)

            # increase labelled lda set
            lda_labelled = np.concatenate((lda_labelled, lda_val[uncertain_samples]), axis=0)

            

            print ('trainset before', X_train.shape, y_train.shape)
            X_train = np.concatenate((X_train, X_val[uncertain_samples]))
            y_train = np.concatenate((y_train, y_val[uncertain_samples]))
            print ('trainset after', X_train.shape, y_train.shape)
            self.samplecount.append(X_train.shape[0])

            bin_count = np.bincount(y_train.astype('int64'))
            unique = np.unique(y_train.astype('int64'))
            print (
                'updated train set:',
                X_train.shape,
                y_train.shape,
                'unique(labels):',
                bin_count,
                unique,
                )
            
            TP_h,TN_h = bin_count[1],bin_count[0]

            X_val = np.delete(X_val, uncertain_samples, axis=0)
            y_val = np.delete(y_val, uncertain_samples, axis=0)
            print ('val set:', X_val.shape, y_val.shape)
            print ()

            lda_val = np.delete(lda_val, uncertain_samples, axis=0)
            print ('lda val set:', lda_val.shape)
            print ()
            print ('labelled lda set:', lda_labelled.shape)
            print ()

            density_val = np.delete(density_val, uncertain_samples, axis=0)
            print ('density val set:', density_val.shape)
            print ()

            

            # normalize again after creating the 'new' train/test sets
            normalizer = Normalize()
            X_train, X_val = normalizer.normalize(X_train, X_val)               

            self.queried += self.k
            
            ## upsampling the data
        
#             ros = RandomOverSampler(random_state=0)
            # ros = SMOTE(random_state=0,k_neighbors=4)
            # X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

            # wihtout upsampling
            X_train_resampled, y_train_resampled = X_train, y_train
            
            bin_count = np.bincount(y_train_resampled.astype('int64'))
            unique = np.unique(y_train_resampled.astype('int64'))
            print('-'*12+'after upsampling'+'-'*12)
            print (
                'updated train set:',
                X_train_resampled.shape,
                y_train_resampled.shape,
                'unique(labels):',
                bin_count,
                unique,
                )
            
            (X_train_re, X_val_re) = self.clf_model.train(X_train_resampled, y_train_resampled, X_val,  'balanced',active_iteration,self.gridsearch_interval,random_seed,self.n_estimators)
            self.clf_model.get_accuracy(y_val,TP_h,TN_h,N,active_iteration)

        print ('final active learning accuracies',
               self.clf_model.accuracies)
        
        print ('final active learning precisions',
               self.clf_model.precisions)
        
        print ('final active learning burdens',
               self.clf_model.burdens)

        print ('final active learning yields',
               self.clf_model.yields)



def experiment(d, models, selection_functions, initial_samples,Ks,max_queried,X,Y,random_seed, gridsearch_interval,doc_lda_matrix, n_estimators):
    algos_temp = []
    print ('stopping at:', max_queried)

    for model_object in models:
        if model_object.__name__ not in d:
            d[model_object.__name__] = {}
      
    for selection_function in selection_functions:

        if selection_function.__name__ == 'QBC':

            model_object = models[1] ##using bagging classifier 
            print('-'*100, 'using ',  model_object.__name__, '-'*100)

            if selection_function.__name__ not in d[model_object.__name__]:
                d[model_object.__name__][selection_function.__name__] = {}

            for k in Ks:
                d[model_object.__name__][selection_function.__name__][str(k)] = {}           

                
                print ('using model = %s, selection_function = %s, k = %s' % (model_object.__name__, selection_function.__name__, k))
                alg = TheAlgorithm(initial_samples,k, 
                                   model_object, 
                                   selection_function,
                                   max_queried,
                                   gridsearch_interval,doc_lda_matrix,
                                   n_estimators
                                   )
                ## ground truth of X,Y of training and test set
                alg.run(X, Y,random_seed)

                d[model_object.__name__][selection_function.__name__][str(k)]['raw_result']=alg.clf_model.raw_result
                # d[model_object.__name__][selection_function.__name__][str(k)]['accuracy']=alg.clf_model.accuracies
                # d[model_object.__name__][selection_function.__name__][str(k)]['precision']=alg.clf_model.precisions
                # d[model_object.__name__][selection_function.__name__][str(k)]['burden']=alg.clf_model.burdens
                # d[model_object.__name__][selection_function.__name__][str(k)]['yield']=alg.clf_model.yields
            
              
                    
                print ()
                print ('---------------------------- FINISHED ---------------------------')
                print ()



        else:
            model_object = models[0] ##using SVM classifier 
            print('-'*100, 'using ',  model_object.__name__, '-'*100)

            if selection_function.__name__ not in d[model_object.__name__]:
                d[model_object.__name__][selection_function.__name__] = {}
        
            for k in Ks:
                d[model_object.__name__][selection_function.__name__][str(k)] = {}           

                
                print ('using model = %s, selection_function = %s, k = %s' % (model_object.__name__, selection_function.__name__, k))
                alg = TheAlgorithm(initial_samples,k, 
                                   model_object, 
                                   selection_function,
                                   max_queried,
                                   gridsearch_interval,doc_lda_matrix,
                                   n_estimators
                                   )
                ## ground truth of X,Y of training and test set
                alg.run(X, Y,random_seed)

                d[model_object.__name__][selection_function.__name__][str(k)]['raw_result']=alg.clf_model.raw_result
                # d[model_object.__name__][selection_function.__name__][str(k)]['accuracy']=alg.clf_model.accuracies
                # d[model_object.__name__][selection_function.__name__][str(k)]['precision']=alg.clf_model.precisions
                # d[model_object.__name__][selection_function.__name__][str(k)]['burden']=alg.clf_model.burdens
                # d[model_object.__name__][selection_function.__name__][str(k)]['yield']=alg.clf_model.yields
            
              
                    
                print ()
                print ('---------------------------- FINISHED ---------------------------')
                print ()

    return d





