import seaborn as sns
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from decision_tree import DT
from classsification_gd import ClassificationGD
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score,roc_auc_score,f1_score,precision_score,balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.utils import resample
from knn import KNN

import matplotlib.pyplot as plt
import time


def update_class(X,main,other):
    result = []

    for i in range(len(X)):
        result.append(1 if X[i] == main else other)
        
    return np.array(result)

def to_multi_label(Y,clases):

    Y_result = []

    for class_ in Y:
        encode_for_elem = [0 for _ in range(len(clases.keys()))]
        encode_for_elem[clases[class_]] = 1
        Y_result.append(encode_for_elem)
    

    return np.array(Y_result)


class multi_svm:
    def __init__(self,alpha,epochs,c=10,batch_size = 30):
        self.models = []
        self.clases = []
        self.alpha = alpha
        self.epochs = epochs
        self.c = c
        self.batch_size = batch_size

   
    def train(self,X_train,Y_train,X_test,Y_test,clases):
        self.models = []
        self.clases = clases 

        for _ in range(len(self.clases)):
            self.models.append(ClassificationGD(self.alpha,self.epochs,self.c,'svm'))
 

        for i in range(len(self.clases)):
            class_ = clases[i]
            Y_train_ = update_class(Y_train,class_,-1)
            Y_test_ = update_class(Y_test,class_,-1)

            self.models[i].train(X_train,Y_train_,X_test,Y_test_,self.batch_size)
    
    def predict(self,X):


        predictions = []


        for point in X:

            scores = {self.clases[i]:self.models[i].svm_raw_predict(point) for i in range(len(self.models))}
            # print(scores)
            
            predictions.append(max(scores,key=scores.get))

        


        return np.array(predictions) 
    def probPredict(self,X):
        return []
    

    def getLosses(self):
        return [model.getLosses() for model in self.models]

class multi_logistic:
    def __init__(self,alpha,epochs,batch_size=30):
        self.models = []
        self.clases = []
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size

   
    def train(self,X_train,Y_train,X_test,Y_test,clases):
        self.models = []
        self.clases = clases 

        for _ in range(len(self.clases)):
            self.models.append(ClassificationGD(self.alpha,self.epochs))
 

        for i in range(len(self.clases)):
            class_ = clases[i]
            Y_train_ = update_class(Y_train,class_,0)
            Y_test_ = update_class(Y_test,class_,0)

            self.models[i].train(X_train,Y_train_,X_test,Y_test_,batch_size=self.batch_size)
    
    def predict(self,X):


        predictions = []


        for point in X:

            scores = {self.clases[i]:self.models[i].logistic_h(point) for i in range(len(self.models))}
            
            # print(scores)
            predictions.append(max(scores,key=scores.get))

        
        return np.array(predictions) 
    

    def probPredict(self,X):
        prob_predictions = []


        for point in X:

            probs = np.array([self.models[i].logistic_h(point) for i in range(len(self.models))])
            probs = probs/np.sum(probs)

            prob_predictions.append(probs)

            

        
        return np.array(prob_predictions) 
    
    def getLosses(self):

        return [model.getLosses() for model in self.models]

    def plot_train_loss(self):
        colors = ['red','green','orange']

        for model,class_ in zip(self.models,range(len(self.clases))):
            epochs,loss = model.get_train_loss()
            plt.plot(epochs,loss,label=f'class {self.clases[class_]}',color=colors[class_])
        # plt.show()
            
    

def k_folds(X,Y,k,model,path=''):

    clases = sorted(list({elem for elem in Y}))
    clases_map = {clases[i]:i for i in range(len(clases))}
        
    n_in_fold = len(Y)//k

    losses = []
    accuracy_ = []
    precision_ = []
    recall_ = []
    f1_score_ = []
    auc_score_ = []
    total_time = 0



    for i in range(k):
        print(i)

        X_train = np.concatenate([X[:i*n_in_fold],X[(i+1)*n_in_fold:]])
        Y_train = np.concatenate([Y[:i*n_in_fold],Y[(i+1)*n_in_fold:]])

        # print(X_train.shape)
        # print(Y_train.shape)


        X_test = X[i*n_in_fold:(i+1)*n_in_fold]
        Y_test = Y[i*n_in_fold:(i+1)*n_in_fold]


        # print(Y_test.shape)

        # print()
        # print()
        # print()
        initial_training_time = time.time()
        model.train(X_train,Y_train,X_test,Y_test,clases)
        training_time = time.time() - initial_training_time
        total_time += training_time 

        print(f'Para el fold {i} se demoro {training_time} para entrenar')

        if path != '' and hasattr(model,'getLosses'):
            losses.append(model.getLosses())

        pred = model.predict(X_test)
        prob = model.probPredict(X_test)

        one_hot_data_pred = to_multi_label(pred,clases_map)

        one_hot_real_data = to_multi_label(Y_test,clases_map)


        # print(confusion_matrix(Y_test,pred)/np.sum(confusion_matrix(Y_test,pred),axis=1)*100)
        print(confusion_matrix(Y_test,pred))
        partial_precision = precision_score(one_hot_real_data,one_hot_data_pred,average=None)
        partial_recall = recall_score(one_hot_real_data,one_hot_data_pred,average=None)
        partial_f1 = f1_score(one_hot_real_data,one_hot_data_pred,average=None)
        partial_auc_score = roc_auc_score(one_hot_real_data,prob,average=None) if len(prob) > 0 else 0
        partial_accuracy = balanced_accuracy_score(Y_test,pred) 

        # model.plot_train_loss()


        print(partial_accuracy,partial_precision,partial_recall,partial_f1,partial_auc_score)

        precision_.append(partial_precision)
        recall_.append(partial_recall)
        f1_score_.append(partial_f1)
        auc_score_.append(partial_auc_score)
        accuracy_.append(partial_accuracy)
        # print(auc(Y_test,predicted_data))
    precision_ = np.sum(np.array(precision_),axis=0)/k
    recall_ = np.sum(np.array(recall_),axis=0)/k
    f1_score_ = np.sum(np.array(f1_score_),axis=0)/k
    auc_score_ = np.sum(np.array(auc_score_),axis=0)/k
    accuracy_ = np.sum(np.array(accuracy_))/k


    print(f'Tiempo total: {total_time}')
    # print(precision_,recall_,f1_score_,auc_score_)

    print('La accuracy es ',accuracy_)
    print('La precision es:',precision_)
    print('El recall es:',precision_)
    print('El f1 score es ',f1_score_)
    print('El auc es ',auc_score_)

    if path != '' and len(losses)>= 1:
        # colors = ['red','green','orange']

        indexes = [i for i in range(model.epochs+1)]
        for i in range(len(clases)):

            type_ = [f'Train losses para k_folds clase {clases[i]}',f'Validation losses para k_folds clase {clases[i]}']
            name_ = ['train','loss']
            for j in range(2):
                for total_losses in losses:
        
                    plt.plot(indexes,total_losses[i][j])

                plt.title(type_[j])
                # plt.show()
                plt.savefig(path+ f'k_folds_{clases[i]}_{name_[j]}.png',format='png')
                plt.clf()
        
        train_average_losses = [np.zeros(model.epochs+1) for _ in range(len(clases)) ]
        val_average_losses = [np.zeros(model.epochs+1) for _ in range(len(clases)) ]

        for i in range(len(clases)):
            for total_loses in losses:
                train_average_losses[i] =  train_average_losses[i] + np.array(total_loses[i][0])
                val_average_losses[i] =  val_average_losses[i] + np.array(total_loses[i][1])

            train_average_losses[i] = train_average_losses[i]/k
            val_average_losses[i] = train_average_losses[i]/k

            # print(train_average_losses[i].shape)
            # print(val_average_losses[i].shape)

            plt.title(f'Loss promedio para k_folds clase {clases[i]}')
            plt.plot(indexes,train_average_losses[i],label='train loss')
            plt.plot(indexes,val_average_losses[i],label='val loss')
            plt.legend()
        
            plt.savefig(path + f'k_folds_{clases[i]}_average.png',format='png')
            plt.clf()

    return [accuracy_,precision_,recall_,f1_score_,auc_score_]




        


def boostrap(X,Y,k,model,path=''):

    clases = sorted(list({elem for elem in Y}))
    clases_map = {clases[i]:i for i in range(len(clases))}
        

    losses = []

    accuracy_ = []
    precision_ = []
    recall_ = []
    f1_score_ = []
    auc_score_ = []
    total_time = 0


    random_states = [42 + i for i in range(k)]


    for i in range(k):
        print(i)


        # while len(Y_test) == 0:

        mask_for_training = np.array([0 for i in range(len(Y))] )


        train_indexes = resample(range(len(Y)),n_samples=(len(Y)),replace=True,random_state=random_states[i])

        mask_for_training[train_indexes] = 1
        mask_for_testing = np.ones(len(Y)) - mask_for_training

        test_indexes = [i for i in range(len(Y)) if mask_for_testing[i]]
        # print(np.sum(mask_for_training))
        # print(np.sum(mask_for_testing))

        
        X_train = X[train_indexes]
        Y_train = Y[train_indexes]


        X_test = X[test_indexes]
        Y_test = Y[test_indexes]


        if len(Y_test) == 0:
            print(f'Para la seed {i} todo el testing es vacio. Continuando...')
            continue



        # print(Y_train.shape)
        # print(Y_test.shape)


        # print()
        # print()
        # print()
        initial_training_time = time.time()
        model.train(X_train,Y_train,X_test,Y_test,clases)
        training_time = time.time() - initial_training_time
        total_time += training_time 

        print(f'Para el boostrap {i} se demoro {training_time} para entrenar')

        if path != '' and hasattr(model,'getLosses'):
            losses.append(model.getLosses())

        pred = model.predict(X_test)
        prob = model.probPredict(X_test)
        one_hot_data_pred = to_multi_label(pred,clases_map)

        one_hot_real_data = to_multi_label(Y_test,clases_map)



        # print(confusion_matrix(Y_test,pred)/np.sum(confusion_matrix(Y_test,pred),axis=1)*100)
        print(confusion_matrix(Y_test,pred))
        partial_precision = precision_score(one_hot_real_data,one_hot_data_pred,average=None)
        partial_recall = recall_score(one_hot_real_data,one_hot_data_pred,average=None)
        partial_f1 = f1_score(one_hot_real_data,one_hot_data_pred,average=None)
        partial_auc_score = roc_auc_score(one_hot_real_data,prob,average=None) if len(prob) > 0 else 0
        partial_accuracy = balanced_accuracy_score(Y_test,pred) 

        # model.plot_train_loss()


        print(partial_accuracy,partial_precision,partial_recall,partial_f1,partial_auc_score)

        precision_.append(partial_precision)
        recall_.append(partial_recall)
        f1_score_.append(partial_f1)
        auc_score_.append(partial_auc_score)
        accuracy_.append(partial_accuracy)
        # print(auc(Y_test,predicted_data))

        # print(auc(Y_test,predicted_data))
    precision_ = np.sum(np.array(precision_),axis=0)/k
    recall_ = np.sum(np.array(recall_),axis=0)/k
    f1_score_ = np.sum(np.array(f1_score_),axis=0)/k
    auc_score_ = np.sum(np.array(auc_score_),axis=0)/k
    accuracy_ = np.sum(np.array(accuracy_))/k


    print(f'Tiempo total: {total_time}')
    # print(precision_,recall_,f1_score_,auc_score_)
    print('La accuracy es ',accuracy_)
    print('La precision es:',precision_)
    print('El recall es:',precision_)
    print('El f1 score es ',f1_score_)
    print('El auc es ',auc_score_)


    if path != '' and len(losses)>= 1:
        # colors = ['red','green','orange']

        indexes = [i for i in range(model.epochs+1)]
        for i in range(len(clases)):

            type_ = [f'Train losses para boostrap clase {clases[i]}',f'Validation losses para boostrap clase {clases[i]}']
            name_ = ['train','loss']
            for j in range(2):
                for total_losses in losses:
        
                    plt.plot(indexes,total_losses[i][j])

                plt.title(type_[j])
                # plt.show()
                plt.savefig(path+ f'boostrap_{clases[i]}_{name_[j]}.png',format='png')
                plt.clf()
        
        train_average_losses = [np.zeros(model.epochs+1) for _ in range(len(clases)) ]
        val_average_losses = [np.zeros(model.epochs+1) for _ in range(len(clases)) ]

        for i in range(len(clases)):
            for total_loses in losses:
                train_average_losses[i] =  train_average_losses[i] + np.array(total_loses[i][0])
                val_average_losses[i] =  val_average_losses[i] + np.array(total_loses[i][1])

            train_average_losses[i] = train_average_losses[i]/k
            val_average_losses[i] = train_average_losses[i]/k

            # print(train_average_losses[i].shape)
            # print(val_average_losses[i].shape)

            plt.title(f'Loss promedio para boostrap clase {clases[i]}')
            plt.plot(indexes,train_average_losses[i],label='train loss')
            plt.plot(indexes,val_average_losses[i],label='val loss')
            plt.legend()
        
            plt.savefig(path + f'boostrap_{clases[i]}_average.png',format='png')
            plt.clf()

    return [accuracy_,precision_,recall_,f1_score_,auc_score_]


def test_hyperparameters_logistic(X,Y,k=10):
    # v = [(0.08,1600),(0.10,1400),(0.15,1000),(0.18,600)]
    v = [(0.15,1600),(0.18,600),(0.5,1600),(0.75,1600),(1,1600),(3,1600),(5,1600),(8,1600),(10,1600)]
    clases_list = sorted(list({elem for elem in data[data.columns[-1]]}))
    clases_dict = {clases_list[i]: i for i in range(len(clases_list))}
    scores = 'alpha,epochs,accuracy,precision,recall,f1,auc\n'

    for method,name in [(k_folds,'k_folds'),(boostrap,'boostrap')]:
        file = open('./hyperparameters_experiments/logistic/' + name + '.csv','w')

        file.write(scores)

        for alpha,epochs in v:
            print(f'Estamos en {alpha} alpha con {epochs} epochs')
            results = [alpha,epochs] +  method(X,Y,k,multi_logistic(alpha,epochs,batch_size=len(Y)))
            # results = [alpha,epochs] + method(X,Y,k,KNN(clases_dict))

            for i in range(len(results)):
                if i < len(results) - 1:
                    file.write(str(results[i]) + ',')
                else:
                    file.write(str(results[i]) + '\n')
        file.close()

            
def test_hyperparameters_svm(X,Y,k=10):
    v = [(0.0000001,1600),(0.000001,1600),(0.00001,1600),(0.0001,1600),(0.001,1600)]
    c = [0.1,1,10,100,1000]
    clases_list = sorted(list({elem for elem in data[data.columns[-1]]}))
    clases_dict = {clases_list[i]: i for i in range(len(clases_list))}
    scores = 'c,accuracy,precision,recall,f1,auc\n'

    for method,name in [(k_folds,'k_folds'),(boostrap,'boostrap')]:
        for alpha,epochs in v:

            file = open('./hyperparameters_experiments/svm/' + name + '_'+ str(alpha)  +'_' + str(epochs) + '.csv','w')

            file.write(scores)

            for c_ in c:

                print(f'Estamos en {alpha} alpha con {epochs} epochs y {c_} c')
                results = [c_] +  method(X,Y,k,multi_svm(alpha,epochs,c=c_,batch_size=len(Y)))
                # results = [c_] + method(X,Y,k,KNN(clases_dict))

                for i in range(len(results)):
                    if i < len(results) - 1:
                        file.write(str(results[i]) + ',')
                    else:
                        file.write(str(results[i]) + '\n')
            file.close()


def test_hyperparameters_knn(X,Y,k=10):
    v = ['l1','l2','chebyshev']
    nn = [3,5,10,20]

    clases_list = sorted(list({elem for elem in data[data.columns[-1]]}))
    clases_dict = {clases_list[i]: i for i in range(len(clases_list))}
    scores = 'nn,accuracy,precision,recall,f1,auc\n'

    for method,name in [(k_folds,'k_folds'),(boostrap,'boostrap')]:
        for distance in v:

            file = open('./hyperparameters_experiments/knn/' + name + '_'+  distance + '.csv','w')

            file.write(scores)

            for n in nn:

                results = [n] + method(X,Y,k,KNN(clases_dict,k=n,distance=distance))

                for i in range(len(results)):
                    if i < len(results) - 1:
                        file.write(str(results[i]) + ',')
                    else:
                        file.write(str(results[i]) + '\n')
            file.close()

            
def test_hyperparameters_dt(X,Y,k=10):

    clases_list = sorted(list({elem for elem in data[data.columns[-1]]}))
    clases_dict = {clases_list[i]: i for i in range(len(clases_list))}
    scores = 'accuracy,precision,recall,f1,auc\n'

    for method,name in [(k_folds,'k_folds'),(boostrap,'boostrap')]:

        file = open('./hyperparameters_experiments/dt/' + name +  '.csv','w')

        file.write(scores)


        results =  method(X,Y,k,DT(clases_dict))

        for i in range(len(results)):
            if i < len(results) - 1:
                file.write(str(results[i]) + ',')
            else:
                file.write(str(results[i]) + '\n')
        file.close()

 


scaler = MinMaxScaler()

data = pd.read_csv('./proyect_dataset/training.csv')
data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])
# data[data.columns[-1]] = update_class(data[data.columns[-1]],1,-1)
# data = data.to_numpy()

print(list({elem for elem in data[data.columns[-1]]}))
# print({elem for elem in update_class(data[data.columns[-1]],1,-1)})
# print(type(data['LB']))
# print(len(data['LB']))


data_train = data.sample(frac=0.7, random_state=43)
data_test = data[~data.index.isin(data_train.index)]




X = data.drop(columns='CLASE').to_numpy()
Y = data['CLASE'].to_numpy()


X_train = data_train.drop(columns='CLASE').to_numpy()
Y_train = data_train['CLASE'].to_numpy()

X_test = data_test.drop(columns='CLASE').to_numpy()
Y_test = data_test['CLASE'].to_numpy()


clases_list = sorted(list({elem for elem in data[data.columns[-1]]}))
clases_dict = {clases_list[i]: i for i in range(len(clases_list))}


# test_hyperparameters_svm(X,Y,10)
# test_hyperparameters_logistic(X,Y,10)
# test_hyperparameters_knn(X,Y,10)
# test_hyperparameters_dt(X,Y,10)

print('Mejores resultados: ')

print('MULTI LOGISTIC')
boostrap(X,Y,10,multi_logistic(8,1600,batch_size=len(Y)))
k_folds(X,Y,10,multi_logistic(5,1600,batch_size=len(Y)))

print('MULTI SVM')
boostrap(X,Y,10,multi_svm(1e-5,1600,c=100,batch_size=len(Y)))
k_folds(X,Y,10,multi_svm(0.0001,1600,c=10,batch_size=len(Y)))

print('DECISION TREE')

boostrap(X,Y,10,DT(clases_dict))
k_folds(X,Y,10,DT(clases_dict))


print('KNN')
boostrap(X,Y,10,KNN(clases_dict,distance='l1'))
k_folds(X,Y,10,KNN(clases_dict,distance='l1'))












