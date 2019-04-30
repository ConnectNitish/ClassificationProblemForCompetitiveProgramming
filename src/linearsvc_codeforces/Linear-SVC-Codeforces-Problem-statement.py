#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import ssl
import copy
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
import re
import random
import pickle
import collections
from pprint import pprint

from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN, SMOTENC, SVMSMOTE
from imblearn.pipeline import make_pipeline

from imblearn.under_sampling import RandomUnderSampler


from sklearn.metrics import hamming_loss, roc_auc_score
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import wordcloud

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer



import scikitplot as skplt
import matplotlib.pyplot as plt

from keras import optimizers
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from scipy.special import softmax

ssl._create_default_https_context = ssl._create_unverified_context


# In[3]:


def clean_statement(statement):
#     x = re.sub('-', ' ', x)
    statement = re.sub('$', ' ', statement)
    statement = re.sub('[^A-Za-z]+', ' ', statement)
    statement = re.sub('[,|.|?|\n]|\t', '', statement)
    statement = re.sub('n\'t', ' ', statement)
    statement = re.sub('submission|submissions|Submission|submission|th ', '', statement)
    statement = re.sub('one|two|given|need', '', statement)
    
    return statement


# In[4]:


def process_problem_statement(q_statement):
    
    q_statement = clean_statement(q_statement)
    
#     q_statement = re.sub('[^A-Za-z]+', ' ', q_statement)
    
    tokens = word_tokenize(q_statement)
    
    stoplist = set(stopwords.words('english'))
    
    word_list = [i for i in q_statement.lower().split() if i not in stoplist]
    
    ps = PorterStemmer()
    
#     word_list = [ps.stem(word) for word in word_list]
    
    q_statement = ' '.join(word_list)
    
#     print(q_statement)
    
    return q_statement


# In[5]:


def process_problem_solution(solution):
    
#     solution = clean_statement(solution)
    
    tokens = word_tokenize(solution)
    
    stoplist = set(stopwords.words('english'))
    
    word_list = [i for i in solution.lower().split() if i not in stoplist]
    
#     ps = PorterStemmer()
    
#     word_list = [ps.stem(word) for word in word_list]
    
    solution = ' '.join(word_list)
    
#     print(q_statement)
    
    return solution


# In[6]:


def process_time_taken(time_col):
#     print(time_col.split())
    return time_col.split()[0]


# In[7]:


def process_tags(all_tags_list,tag_col):
    
#     print(tag_col)
    tags_present = list(re.split(',',tag_col))
    return tags_present


# In[8]:


def get_all_distinct_tags(tags_col):
    
    tags_list = []
    
    t_sets = set(tags_list)
    
    for row in tags_col:
#         print(row)
        t_list = re.split(',',row)
#         print(t_list)
        t_sets = t_sets.union(set(t_list))
#         print(t_sets)
    tags_list = list(t_sets)
    
    stoplist = set(stopwords.words('english'))
    
    word_list = [i for i in tags_list if i not in stoplist]
    
    return tags_list


# In[9]:


# tag list obtained from the dataset
# global tags_list

tags_list = ['dsu', 'trees', 'chinese remainder theorem', 'sortings', 'games', 'implementation', 'bitmasks',
              '*special', 'hashing', 'geometry', 'two pointers', 'combinatorics', 'flows', 'strings',
              'probabilities', 'data structures', 'ternary search', 'greedy', 'math', 'matrices',
              'divide and conquer', 'dfs and similar', 'constructive algorithms', 'brute force', 'dp',
              '2-sat', 'graph matchings', 'binary search', 'number theory', 'graphs', 'fft', 'shortest paths',
              'schedules', 'meet-in-the-middle', 'string suffix structures', 'expression parsing']



# In[10]:


def plot_class_distribution(Y,classes):
    
    count_list = [0]*Y.shape[1]
    
    for index in range(Y.shape[1]):
        
        count_list[index] = np.sum(Y[:,index])/Y.shape[0]
    
    plt.figure(figsize=(8, 8), dpi=100)
    
    plt.barh(classes,count_list, align='center', alpha=0.5)
#     plt.bar(np.arange(Y.shape[1]),count_list, align='center', alpha=0.5)
#     plt.plot(np.arange(Y.shape[1]),count_list)
    plt.show()
    


# In[39]:


def data_preprocessing():
    
    global X, Y, mlb, distinct_tags
    
    df = pd.read_csv("codeforces_question_v6.csv")

    print(df.columns)
    
    print(df.shape)
    df = df.drop_duplicates()
    print(df.shape)
    
    distinct_tags = get_all_distinct_tags(df["tags"])
    
    df["problem statement"] = [process_problem_statement(x) for x in df["problem statement"]]
    
    X = copy.deepcopy(df["problem statement"])
    Y = [process_tags(distinct_tags,x) for x in df["tags"]]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y)
    
    plot_class_distribution(Y,mlb.classes_)


# In[40]:


data_preprocessing()


# In[41]:


def reduce_sample_imbalance(X_train,Y_train):
    
    global X, Y, mlb, distinct_tags
    
    X_train = pd.DataFrame(X_train).values
    
    xtrain_col_index = X_train.shape[1]
    xtrain_row_index = X_train.shape[0]
    
    XY = np.column_stack((X_train,Y_train))
    
    labels = [0]*len(mlb.classes_)
    row_index_to_delete = []
    ratio = []
    
    for index in range(Y_train.shape[1]):
        ratio.append(np.sum(Y_train[:,index])/Y_train.shape[0])
        if np.sum(Y_train[:,index])/Y_train.shape[0] > 0.03:
            
            labels[index] = 1
            
    ratio.sort()

    count_deletions = [0]*len(labels)
    for index in range(len(labels)):
        if labels[index] == 1:

            count_deletions[index] = int(np.sum(Y_train[:,index]) * 0.97)    
    
    for index in range(Y_train.shape[0]):
        flag_to_delete = True
        
        for yindex in range(X_train.shape[1],XY.shape[1]):
            if Y_train[index][yindex-X_train.shape[1]] == 1:
                if labels[yindex-X_train.shape[1]] == 0 or count_deletions[yindex-X_train.shape[1]] <= 0:
                    flag_to_delete = False
                    break
        
        if flag_to_delete:
            row_index_to_delete.append(index)
            for i in range(len(count_deletions)):
                count_deletions[i] -= Y_train[index][i]
    
        if sum(count_deletions) <= 0:
            break;
            
    XY = np.delete(XY, row_index_to_delete,axis=0)
    
    ratio = []
    for index in range(X_train.shape[1],XY.shape[1]):
        ratio.append(np.sum(XY[:,index])/XY.shape[0])
    
    plot_class_distribution(XY[:,X_train.shape[1]:],mlb.classes_)
    
    
    data_to_repeat_index = [1]*XY.shape[0]
    labels = [0]*len(mlb.classes_)
    ratio = []
    for index in range(xtrain_col_index,XY.shape[1]):
        ratio.append(np.sum(XY[:,index])/XY.shape[0])
        if np.sum(XY[:,index])/XY.shape[0] < 0.02:
            labels[index-xtrain_col_index] = 1
    

    ratio.sort()
        
    for index in range(XY.shape[0]):
        for yindex in range(X_train.shape[1],XY.shape[1]):
            if XY[index][yindex] == 1 and labels[yindex-X_train.shape[1]] > 0:
                data_to_repeat_index[index] = 100
                labels[yindex-X_train.shape[1]] = -1
    
    XY = np.repeat(XY, repeats = data_to_repeat_index, axis=0)

    X_train = XY[:,0]
    Y_train = XY[:,1:]
            
    X_train = pd.Series(X_train)
    X_train = X_train.astype(str)
    Y_train = Y_train.astype('int') 
    return X_train,Y_train


# In[42]:


def oversample_train_data(X_train,Y_train):
    
    global X, Y, mlb, distinct_tags
    
    X_train = pd.DataFrame(X_train).values
    
    XY = np.column_stack((X_train,Y_train))
    
    labels = []
    data_to_repeat_index = [1]*Y_train.shape[0]
    
    for index in range(Y_train.shape[1]):
        if np.sum(Y_train[:,index])/Y_train.shape[0] < 0.02:
            labels.append(index)
    
    for index in range(Y_train.shape[0]):
        for label in labels:
            if Y_train[index][label] == 1:
                data_to_repeat_index[index] = 200
    

    XY = np.repeat(XY, repeats = data_to_repeat_index, axis=0)
    
    X_train = XY[:,0]
    Y_train = XY[:,1:]
        
#     Ytrain = mlb.inverse_transform(Y_train)
    
#     print(Ytrain)
    
    X_train = pd.Series(X_train)
    X_train = X_train.astype(str)
    Y_train = Y_train.astype('int') 
    return X_train,Y_train


# In[47]:


global X, Y, mlb, distinct_tags

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.2, random_state = 0)

plot_class_distribution(Y,mlb.classes_)
# X_train,Y_train = reduce_sample_imbalance(X,Y)
# plot_class_distribution(Y,mlb.classes_)
plot_class_distribution(Y,mlb.classes_)


classifier = make_pipeline(
    CountVectorizer(ngram_range = (1,2),binary = True,lowercase=False),
    TfidfTransformer(norm = 'l2',sublinear_tf = True),
    OneVsRestClassifier(LinearSVC(penalty="l2",loss="squared_hinge",tol=1,random_state=0,max_iter=1000,C = 1)))

classifier.fit(X_train, Y_train)
predicted = classifier.predict(X_train)


# In[48]:



print(Y_train.shape)
print(predicted.shape)
y_labels_predicted = mlb.inverse_transform(predicted)
y_labels_actual = mlb.inverse_transform(Y_train)

print("On Train data")
print("hamming_loss: ",hamming_loss(Y_train,predicted))
print("recall_score: ",recall_score(Y_train,predicted,average = 'weighted'))
print("precision_score: ",precision_score(Y_train,predicted,average = 'weighted'))
print("f1_score: ",f1_score(Y_train,predicted,average = 'weighted'))

print("roc_auc_score: ",roc_auc_score(Y_train,predicted,average = 'weighted'))

print()
print()

# print("Actual vs Predicted")

# for item, labels in zip(y_labels_actual, y_labels_predicted):
#         print('{0} => {1}'.format(item, ', '.join(labels)))

# print()
# print()


print("On Validation data")
predicted = classifier.predict(X_validation)

y_labels_predicted = mlb.inverse_transform(predicted)
y_labels_actual = mlb.inverse_transform(Y_validation)
# print(predicted)
print(Y_validation.shape)
Y_validation[int(Y_validation.shape[0]/2),:] =  1
print(predicted.shape)
print("hamming_loss: ",hamming_loss(Y_validation,predicted))
print("recall_score: ",recall_score(Y_validation,predicted,average = 'weighted'))
print("precision_score: ",precision_score(Y_validation,predicted,average = 'weighted'))
print("f1_score: ",f1_score(Y_validation,predicted,average = 'weighted'))


print()
print()

print("roc_auc_score: ",roc_auc_score(Y_validation,predicted,average = 'weighted'))
# print("confusion_matrix: ",confusion_matrix(Y_validation,predicted))

print()
print()
print("Actual vs Predicted")

for item, labels in zip(y_labels_actual, y_labels_predicted):
        print('{0} => {1}'.format(item, ', '.join(labels)))


# for col_index in range(Y_validation.shape[1]):
#     print(confusion_matrix(Y_validation[:,col_index],predicted[:,col_index]))
    
labels_list = mlb.classes_
roc_auc_list = []
for col_index in range(Y_validation.shape[1]):
    roc_auc_list.append(roc_auc_score(Y_validation[:,col_index],predicted[:,col_index]))

plt.figure(figsize=(8, 8), dpi=100)
plt.barh(labels_list,roc_auc_list, align='center', alpha=0.5)
plt.show()

################################################################################

metrics_list = []
recall_list = []
precision_list = []
roc_auc_list = []
f1_score_list = []

classes_list = mlb.classes_
    
for col_index in range(Y_validation.shape[1]):
    recall_list.append(recall_score(Y_validation[:,col_index],predicted[:,col_index]))
    
for col_index in range(Y_validation.shape[1]):
    precision_list.append(precision_score(Y_validation[:,col_index],predicted[:,col_index]))
    
for col_index in range(Y_validation.shape[1]):
    f1_score_list.append(f1_score(Y_validation[:,col_index],predicted[:,col_index]))
    
for col_index in range(Y_validation.shape[1]):
    roc_auc_list.append(roc_auc_score(Y_validation[:,col_index],predicted[:,col_index]))

metrics_list = [mlb,classes_list,recall_list,precision_list,f1_score_list,roc_auc_list,Y_validation,predicted]

with open('Linear_SVC_Codeforces', 'wb') as fp:
    pickle.dump(metrics_list, fp)

################################################################################



# In[ ]:





# In[ ]:




