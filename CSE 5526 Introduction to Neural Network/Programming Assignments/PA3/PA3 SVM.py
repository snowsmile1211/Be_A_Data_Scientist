# CSE 5526 Programming Assignment 3 SVM #
# There are two parts in this project
# When you execute part 1, you can comment part 2 and vice versa
from libsvm.python.svm import *
from libsvm.python.svmutil import *
import random
import numpy as np

y_train, x_train = svm_read_problem('ncrna_s.train.txt')
num_train=len(y_train)
y_test, x_test = svm_read_problem('ncrna_s.test.txt')
num_test=len(y_test)
print('training set:',num_train)
print(y_train[0])
print('testing set:',num_test)
print(x_train[0])

c_list=[2**(i-4) for i in range(0,13)]
alpha_list=[2**(i-4) for i in range(0,13)]

# Part 1: Classification using linear SVMs
prob  = svm_problem(y_train, x_train)
for c in c_list:
    print('value of c is: ',c)
    param = svm_parameter('-t 0 -h 0 -c '+str(c))
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
    print('value of c is: ',c)

# Part 2: Classification using RBF kernel SVM

def calculate_acc(cv_train,cv_test,param,num_subset):
    
    y_cv_train=[cv_train[i][0] for i in range(num_subset*4)]
    x_cv_train=[cv_train[i][1] for i in range(num_subset*4)]
    
    y_cv_test=[cv_test[i][0] for i in range(num_subset)]
    x_cv_test=[cv_test[i][1] for i in range(num_subset)]
    prob = svm_problem(y_cv_train,x_cv_train)
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y_cv_test, x_cv_test, m)
    return p_acc[0]/100

data_pair=[(y_train[i],x_train[i]) for i in range(num_train)]
random.shuffle(data_pair)
num_cv=int(num_train/2)
CV_set=data_pair[:num_cv]
num_subset=int(num_cv/5)
CV_set_y=CV_set
CV_set_1=CV_set[:num_subset]
CV_set_2=CV_set[num_subset:num_subset*2]
CV_set_3=CV_set[num_subset*2:num_subset*3]
CV_set_4=CV_set[num_subset*3:num_subset*4]
CV_set_5=CV_set[num_subset*4:num_subset*5]
ave_acc_mat=[]
for c in c_list:
    ave_acc_list=[]
    for alpha in alpha_list:
        param = svm_parameter('-t 2 -h 0 -g '+str(alpha)+' -c '+str(c))
        acc_list=[]
        #set 1
        cv_train=CV_set_2+CV_set_3+ CV_set_4 + CV_set_5
        cv_test=CV_set_1
        acc1=calculate_acc(cv_train,cv_test,param,num_subset)
        acc_list.append(acc1)
        #set 2
        cv_train=CV_set_1+CV_set_3+ CV_set_4 + CV_set_5
        cv_test=CV_set_2
        acc1=calculate_acc(cv_train,cv_test,param,num_subset)
        acc_list.append(acc1)
        #set 3
        cv_train=CV_set_2+CV_set_1+ CV_set_4 + CV_set_5
        cv_test=CV_set_3
        acc1=calculate_acc(cv_train,cv_test,param,num_subset)
        acc_list.append(acc1)
        #set 4
        cv_train=CV_set_2+CV_set_3+ CV_set_1 + CV_set_5
        cv_test=CV_set_4
        acc1=calculate_acc(cv_train,cv_test,param,num_subset)
        acc_list.append(acc1)
        #set 5
        cv_train=CV_set_2+CV_set_3+ CV_set_4 + CV_set_1
        cv_test=CV_set_5
        acc1=calculate_acc(cv_train,cv_test,param,num_subset)
        acc_list.append(acc1)
        ave_acc=sum(acc_list)/len(acc_list)
        ave_acc_list.append(ave_acc)
    ave_acc_mat.append(ave_acc_list)

print('\n got the index\n')
acc_mat=np.array(ave_acc_mat).reshape(13,13)
print(acc_mat)
index=np.argmax(acc_mat)
row=index//13
column=index%13
C=c_list[row]
alpha=alpha_list[column]
print('\n predict with the whole training set\n')
prob  = svm_problem(y_train, x_train)
param = svm_parameter('-t 2 -h 0 -g '+str(alpha)+' -c '+str(c))
m = svm_train(prob, param)
p_label, p_acc, p_val = svm_predict(y_test, x_test, m)

