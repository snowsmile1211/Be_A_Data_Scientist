#### CSE 5526 Neural Networks Programming Assignment 1  ####

### import basic package in python ###
import random
import math
import numpy as np
### Define the classes needed for the project ###
class InputNode:
    def __init__(self,data,WeightList):
        # data is the input data
        # weightlist is the weight for each of the next layer node
        self.data=data
        self.WeightList=WeightList
    def __getdata__(self): # get the input data of this input node
        return self.data
    def __getWeight__(self,i): 
        # get weight between the node and ith node in next layer
        return self.WeightList[i]

class HiddenNode:
    # for this project, there is only one hidden layer
    # no need to specify the index
    def __init__(self,bias,WeightList):
        self.bias=bias
        self.WeightList=WeightList
    def __getbias__(self): # get the input data of this input node
        return self.bias
    def __getWeight__(self,i): 
        # get weight between the node and ith node in next layer
        return self.WeightList[i]

class OutputNode:
    def __init__(self,bias,output=None):
        self.bias=bias
        self.output=output
    def __getbias__(self): # get the input data of this input node
        return self.bias
    def __getoutput__(self): 
        # get weight between the node and ith node in next layer
        return self.output

### Main Program ###
# Definition of the activation function (logistic sigmoid)
def LogFunc(x):
    try:
        ans = 1.0/(1+math.exp(-x))
    except OverflowError:
        if x <-100:
            return 0
        elif x>100:
            return 1
    return ans
# Definition of the derivative of activation function
def LogFuncDer(x):
    derivative=LogFunc(x)-LogFunc(x)**2
    return derivative
# calculate the weighted sum for each of the nodes
# each of the input parameter is an numpy array
def WeightedSum(input_vec,weight_vec):
    if len(input_vec)!=len(input_vec):
        print("The two input vector should have same length")
        return
    else:
        sum=np.dot(input_vec,weight_vec)
        return sum
# Definition of cost function, in this project
# there is only one output node
def CostFunction(act_output,exp_output):
    ESquare=((act_output-exp_output)**2)*0.5
    return ESquare
# Function for calculating deltaE for output layer
# output is also a array(same as input_vec 4*1) of deltaE for each of the weight
def DeltaE_op(input_vec,weight_vec,exp_output):
    # inputlist is the input for output layer
    vk=WeightedSum(input_vec,weight_vec)
    act_output=LogFunc(vk)
    ek=exp_output-act_output
    phi_prime=LogFuncDer(vk)
    deltaE=input_vec*ek*phi_prime
    return deltaE

# Initialize the neural network
InputNodeList=[]
HiddenNodeList=[]
num_hiddennode=4
num_inputnode=4
num_outputnode=1
randnum=random.uniform(-1,1)
#generate the input and output array
# input_vec_mat=np.array([[1,1,2,3],[1,2,3,4],[2,1,1,1],[3,4,5,6],[][][][][][]]) # a sample input list for programming
# exp_output=np.array([0,1,1,0])
num_sample=16
# input_vec_mat=np.array([[random.randint(0,9)for i in range(4)] for j in range(num_sample)])
# exp_output=np.array([np.count_nonzero(input_vec_mat[i]==1)%2 for i in range(num_sample)])
input_vec_mat=np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],
                        [1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]])
exp_output=np.array([np.count_nonzero(input_vec_mat[i]==1)%2 for i in range(num_sample)])
# exp_output=np.array([0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0])
# input_vec=np.insert(input_vec_mat,0,1)

# input_vec=np.insert(input_vec_mat[0],0,1)
# bias of hidden layer should be appended to the weight matrix
weight_matrix_in2hi=np.array([[random.uniform(-1,1) for i in range(num_hiddennode)] for j in range(num_inputnode)])
weight_vec_hi2op=np.array([random.uniform(-1,1) for i in range(num_hiddennode)]).reshape(4,1)
bias_hi=np.array([random.uniform(-1,1) for i in range(num_hiddennode)]).reshape(4,1)
bias_op=random.uniform(-1,1)
ek=np.ones([16])
# Learning rate gama should vary from 
gama=0.5
index_iter=0

while True:
    np.random.shuffle(input_vec_mat)
    exp_output=np.array([np.count_nonzero(input_vec_mat[i]==1)%2 for i in range(num_sample)])
    input_vec=input_vec_mat[index_iter%num_sample].reshape(4,1)
    vj=np.dot(weight_matrix_in2hi,input_vec)+bias_hi # vj is also a vector
    yj=np.array([LogFunc(v) for v in vj]).reshape(4,)
    
    # vk=WeightedSum(yj,weight_vec_hi2op)
    vk=np.sum(weight_vec_hi2op*yj)+bias_op
    act_output=LogFunc(vk)
    ek_temp=exp_output[index_iter%num_sample]-act_output
    ek[index_iter%num_sample]=ek_temp

    phi_prime_vk=LogFuncDer(vk)
    phi_prime_vj=np.array([LogFuncDer(v) for v in vj]).reshape(4,1)
    # DeltaE_in2hi=np.empty([num_inputnode+1,num_hiddennode])
    # theta_j=np.ones([num_inputnode+1,num_hiddennode])
    Dweight_vec_hi2op=(-ek_temp*phi_prime_vk*yj).reshape(4,1)
    Dbias_op=-ek_temp*phi_prime_vk*1

    delta=-(ek_temp)*phi_prime_vk
    Dweight_matrix_in2hi = np.outer(np.multiply(phi_prime_vj,  delta*weight_vec_hi2op), input_vec)
    Dbias_hi = np.outer(np.multiply(phi_prime_vj,  delta*weight_vec_hi2op), 1)

    weight_matrix_in2hi=weight_matrix_in2hi-gama*Dweight_matrix_in2hi
    bias_hi=bias_hi-gama*Dbias_hi
    weight_vec_hi2op=weight_vec_hi2op-gama*Dweight_vec_hi2op
    bias_op=bias_op-gama*Dbias_op

    # DeltaE_hi2op=-ek[index_iter%num_sample]*phi_prime_vk*yj
    
    # delta=ek[index_iter%num_sample]*phi_prime_vk
    # dw_in2hi=np.outer(np.multiply(phi_prime_vj,delta*weight_vec_hi2op[1:5]),input_vec) # some problems about this equation
    
    # weight_vec_hi2op=weight_vec_hi2op-DeltaE_hi2op*gama
    # weight_matrix_in2hi=weight_matrix_in2hi-dw_in2hi*gama
 
    index_iter=index_iter+1

    if index_iter%num_sample==0:
        E=0
        for i in range (num_sample):
            input_vec=input_vec_mat[i].reshape(4,1)
            vj=np.dot(weight_matrix_in2hi,input_vec)+bias_hi # vj is also a vector
            yj=np.array([LogFunc(v) for v in vj]).reshape(4,)
            # vk=WeightedSum(yj,weight_vec_hi2op)
            vk=np.sum(weight_vec_hi2op*yj)+bias_op
            act_output=LogFunc(vk)
            E=E+CostFunction(exp_output[i],act_output)
        # print("Index of epoch:"+str(int(index_iter/num_sample)))
        print("Index of epoch:"+str(int(index_iter/num_sample))+" Absolute error:" + str(E))
        is_abs_err_small=abs(ek)<=0.05
        if not (False in is_abs_err_small):
            print("Index of iteration:"+str(index_iter))
            print(ek)
            break
    

    

    
