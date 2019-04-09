#### CSE 5526 Neural Networks Programming Assignment 1  ####

### import basic package in python ###
import random
import math
import numpy as np
import itertools
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
        
### Define the functions needed for the project ###

# activation function
def LogFunc(x):
    try:
        ans = 1.0/(1+math.exp(-x))
    except OverflowError:
        if x <-100:
            return 0
        elif x>100:
            return 1
    return ans

# derivative of activation function
def LogFuncDer(x):
    return LogFunc(x) * (1 - LogFunc(x))

# Cost/loss function
def CostFunction(act_output,exp_output):
    ESquare=((act_output-exp_output)**2)*0.5
    return ESquare

# get the expected output based on the input data
def Exp_output(input_vec):
    input_vec = input_vec.reshape(4,)
    exp_output = list(input_vec.reshape(4,)).count(1) % 2
    return exp_output

# BackPropogation between output layer and hidden layer
def BackProp_op2hi(yj, w_hi2op, exp_output, act_output):
    vk = sum(np.multiply(w_hi2op.reshape(4,), yj.reshape(4,)))
    dw_hi2op = -(exp_output-act_output) * LogFuncDer(vk+b_op) * yj
    db_op =  -(exp_output-act_output) * LogFuncDer(vk + b_op) * 1
    return (dw_hi2op, db_op)

# BackPropogation between hidden layer and input layer
def BackProp_hi2in(input_vec, w_in2hi, b_hi, yj, w_hi2op):
    vj = np.dot(w_in2hi, input_vec) + b_hi
    vj_prime = np.array([-LogFuncDer(vj[0]), -LogFuncDer(vj[1]), -LogFuncDer(vj[2]), -LogFuncDer(vj[3])]).reshape(4,1)
    vk = sum(np.multiply(w_hi2op.reshape(4,), yj.reshape(4,)))
    delta = (exp_output-act_output) * LogFuncDer(vk + b_op)
    dw_in2hi = np.outer(np.multiply(vj_prime,  delta*w_hi2op), input_vec)
    db_hi = np.outer(np.multiply(vj_prime,  delta*w_hi2op), 1)
    assert np.shape(dw_in2hi) == np.shape(w_in2hi) == (4,4)
    assert np.shape(db_hi) == np.shape(b_hi) == (4,1)
    return (dw_in2hi, db_hi)

# The whole backpropogation
def BackProp(input_vec, act_output, w_in2hi, b_hi, w_hi2op, b_op):
    (_, yj) = Forward_in2hi(input_vec, w_in2hi, b_hi)
    act_output = Forward_hi2op(w_hi2op, b_op, yj)
    Forward(input_vec, w_in2hi, w_hi2op, b_hi, b_op)
    (dw_hi2op, db_op) = BackProp_op2hi(yj, w_hi2op, exp_output, act_output)
    (dw_in2hi, db_hi) = BackProp_hi2in(input_vec, w_in2hi, b_hi, yj, w_hi2op)
    return (dw_in2hi, dw_hi2op, db_hi, db_op)

# Forward Process between input and hidden layer
def Forward_in2hi(input_vec, w_in2hi, b_hi):
    input_vec=input_vec.reshape(4,1)
    tmp = np.dot(w_in2hi, input_vec) + b_hi
    tmp = tmp.reshape(4,)
    result = [LogFunc(tmp[0]), LogFunc(tmp[1]), LogFunc(tmp[2]), LogFunc(tmp[3])]
    result = np.array(result).reshape(4,1)
    return (tmp, result)

# Forward Process between hidden and output layer
def Forward_hi2op(w_hi2op, b_op, yj):
    yj=yj.reshape(4,1)
    w_hi2op = w_hi2op.reshape(4,)
    yj = yj.reshape(4,)
    tmp = np.sum(w_hi2op * yj) + b_op
    return LogFunc(tmp)

# The whole forward process
def Forward(input_vec, w_in2hi, w_hi2op, b_hi, b_op):
    (_, yj) = Forward_in2hi(input_vec, w_in2hi, b_hi)
    act_output = Forward_hi2op(w_hi2op, b_op, yj)
    return act_output

# Update the weights based on several parameters
def Weight_update(w_in2hi, b_hi, w_hi2op, b_op, dw_in2hi, dw_hi2op, db_hi, db_op, dw_in2hi_, dw_hi2op_, db_hi_, db_op_, gama=0.05,alpha=0.9):
    dw_in2hi = gama * dw_in2hi + alpha * dw_in2hi_
    dw_hi2op = gama * dw_hi2op + alpha * dw_hi2op_
    db_hi = gama * db_hi + alpha* db_hi_
    db_op = gama * db_op + alpha * db_op_
    return (w_in2hi-dw_in2hi, b_hi-db_hi, w_hi2op-dw_hi2op, b_op- db_op, dw_in2hi, db_hi, dw_hi2op, db_op)

# Form the format of input and output data pair following the defined order
def Data_pair():
    data = []
    for input_vec in itertools.product([0, 1], [0, 1], [0, 1], [0, 1]):
        input_vec = np.array(input_vec)
        data +=[(input_vec.reshape(4,1), Exp_output(input_vec))]
    return data

# Form the format of input and output data pair randomly
def Data_pair_shuffle():
    data = []
    for input_vec in itertools.product([0, 1], [0, 1], [0, 1], [0, 1]):
        input_vec = np.array(input_vec)
        data +=[(input_vec.reshape(4,1), Exp_output(input_vec))]
    random.shuffle(data)
    return data

def initialize_gradient():
    dw_in2hi = np.zeros((4, 4))
    db_hi = np.zeros((4, 1))
    dw_hi2op = np.zeros((4, 1))
    db_op = 0
    return (dw_in2hi, db_hi, dw_hi2op, db_op)



###          Main program              ###
# Initialize the data_pair which is used for error test
data_pair=Data_pair()
gama_list=[(0.5-0.05*i) for i in range(10)]
for gama in gama_list:

    np.random.seed(42)
    input_vec = np.random.choice([0,1],4).reshape(4,1)
    w_in2hi = np.random.rand(4,4)*2-1
    b_hi = np.random.rand(4,1)*2-1
    w_hi2op = np.random.rand(4,1)*2-1
    b_op = np.random.rand()*2-1
    exp_output = Exp_output(input_vec)
    
    new_update = initialize_gradient()
    index_epoch=0
    while True:
        cost_total = 0
        dw_in2hi = np.zeros((4, 4))
        db_hi = np.zeros((4, 1))
        dw_hi2op = np.zeros((4, 1))
        db_op = 0
        data_pair_shuffle=Data_pair_shuffle()
        # randomize the order of input patterns for online learning
        for (input_vec, exp_output) in data_pair_shuffle:
            act_output = Forward(input_vec, w_in2hi, w_hi2op, b_hi, b_op)
            (dw_in2hi, dw_hi2op, db_hi, db_op) = BackProp(input_vec, exp_output, w_in2hi, b_hi, w_hi2op, b_op)
            old_update = new_update
            (dw_in2hi_, db_hi_, dw_hi2op_, db_op_) = old_update
            # determine the value of alpha, when alpha is 0.9, the learning rate is controlled by momentum
            # when alpha is 0, the learning rate is not controlled by momentum
            (w_in2hi, b_hi, w_hi2op, b_op, dw_in2hi, db_hi, dw_hi2op, db_op) = Weight_update(w_in2hi, b_hi, w_hi2op, b_op, dw_in2hi, dw_hi2op, db_hi, db_op, dw_in2hi_, dw_hi2op_, db_hi_, db_op_, gama=gama,alpha=0)
            new_update=(dw_in2hi, dw_hi2op, db_hi, db_op) 
        # calculate the value of cost function for each of the input pattern.
        ek=np.array([])
        for (input_vec, exp_output) in data_pair:
            act_output = Forward(input_vec, w_in2hi, w_hi2op, b_hi, b_op)
            cost_total += CostFunction(act_output, exp_output)
            ek=np.append(ek,[exp_output-act_output])

        index_epoch=index_epoch+1

        is_abs_err_small=abs(ek)<=0.05
        if not (False in is_abs_err_small):
            print('Learning rate is %s, number of epochs %s\n '%(gama, index_epoch))
            print(ek)
            break

        # if index_epoch % 100 == 0:
        #     print('Epoch is %s: Total Loss is %s' % (int(index_epoch), cost_total))
        # if cost_total < 0.01:
        #     print ('Learning rate is %s, number of epochs %s\n '%(gama, index_epoch))
        #     print(ek)
        #     break


