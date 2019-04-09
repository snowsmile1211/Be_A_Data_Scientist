
#####      CSE 5526 Introduction to Neural Networks       #####
#####          Programming Assignment 2    RBF            #####
#####                   Online Learning                    #####

import numpy as np
import matplotlib.pyplot as plt

class Cluster:
    def __init__(self,data,num_elements):
        self.data=data
        self.num_elements=num_elements
    def __getdata__(self): # get the input data of this input node
        return self.data
        
def find_cluster(x,center_list,num_cluster):
    # Find the index of cluster in which x is 
    # Besides the index of cluster, also return the distance list to each of the centers
    dist_list=[]
    for i in range(num_cluster):
        dist=abs(x-center_list[i])
        dist_list.append(dist)
    min_dist=min(dist_list)
    index_cluster=dist_list.index(min_dist)
    return (index_cluster,dist_list)

def Kmeans(k,input_x,sample_size):
    # function for K-means #
    # input parameter: k number of cluster centers
    # input_x input patterns for K-means
    # output: a list of cluster centers

    #Initialize centers randomly
    # index_center_list=np.random.randint(sample_size,size=k)
    a = np.arange(75)
    np.random.shuffle(a)
    index_center_list=a[0:k]
    center_list=np.array([input_x[index_center] for index_center in index_center_list]).reshape(k,)
    
    index_cluster_sample=np.ones(sample_size)*(-1)
    
    while True:
        # calculate the index of cluster for each of the input patterns
        for i in range(sample_size):
            (index_cluster_sample[i],dist_test)=find_cluster(input_x[i],center_list,k)
            pass
    
        center_list_new=np.ones(k,)*(-99)
        # Update cluster centers
        for j in range(k):
            sum_x=0
            count_j=0
            for i in range(sample_size):
                if index_cluster_sample[i]==j:
                    sum_x=sum_x+input_x[i]
                    count_j=count_j+1
            center_list_new[j]=sum_x/count_j
        dif0=abs(center_list_new[0]-center_list[0])
        dif1=abs(center_list_new[1]-center_list[1])
        print('dif0 %s, dif1 %s\n '%(dif0, dif1))
        if np.array_equal(center_list_new,center_list):
            break
        center_list=center_list_new
    return center_list_new

def max_distance(center_list,k):
    dmax=0
    for i in range(k):
        for j in range(k):
            if i!=j:
                d=abs(center_list[i]-center_list[j])
                if d>dmax:
                    dmax=d
                    pass
                pass
            pass
        pass
    return dmax

# Activation functions
def gaussian(x,xj,delta):
    dif_2=(x-xj)**2
    phi=np.exp(-dif_2/(2*delta**2))
    return phi
    
# Derivative of activation function
def gaussianDer(x,xj,delta):
    coef=-abs(x-xj)/(delta**2)
    dif_2=(x-xj)**2
    expv=-dif_2/(2*delta**2)
    phi_prime=coef*np.exp(expv)
    return phi_prime

# Forward Process between input and hidden layer
def Forward_in2hi(input_x, xj_list,delta,k):
    result = [gaussian(input_x,xj_list[i],delta) for i in range(k)]
    result = np.array(result).reshape(k,1)
    return result

# Forward Process between hidden and output layer
def Forward_hi2op(w_hi2op, b_op, yj,k):
    yj=yj.reshape(k,1)
    w_hi2op = w_hi2op.reshape(k,)
    yj = yj.reshape(k,)
    tmp = np.sum(w_hi2op * yj) + b_op
    return (tmp)

# The whole forward process
def Forward(input_x, xj_list,delta, w_hi2op, b_op,k):
    yj = Forward_in2hi(input_x, xj_list,delta,k)
    act_output = Forward_hi2op(w_hi2op, b_op, yj,k)
    return act_output

# BackPropogation between output layer and hidden layer
def BackProp_op2hi(input_x, xj_list,w_hi2op, b_op, exp_output, act_output,k):
    yj= Forward_in2hi(input_x,xj_list,delta, k)
    act_output = Forward_hi2op(w_hi2op, b_op, yj,k)
    # vk = sum(np.multiply(w_hi2op.reshape(k,), yj.reshape(k,)))
    dw_hi2op = -(exp_output-act_output) * yj
    dw_hi2op=np.array(dw_hi2op).reshape(k,1)
    db_op =  -(exp_output-act_output) * 1
    return (dw_hi2op, db_op)

# Update the weights based on several parameters
def Weight_update( w_hi2op, b_op , dw_hi2op, db_op, gama=0.02):
    dw_hi2op = gama * dw_hi2op
    db_op = gama * db_op
    return (w_hi2op-dw_hi2op,b_op- db_op,dw_hi2op, db_op)
    
# Cost/loss function
def CostFunction(act_output,exp_output):
    ESquare=((act_output-exp_output)**2)*0.5
    return ESquare

# Initialize the input and output data for training samples
sample_size=75
np.random.seed(42)
noise=np.random.uniform(-0.1,0.1,sample_size).reshape(sample_size,1)
input_x=np.random.uniform(0,1,sample_size).reshape(sample_size,1)
output_h=0.4*np.sin(2*np.pi*input_x)+0.5+noise
# print(output_h.reshape(1,sample_size))

# ax = plt.gca()
# ax.scatter([input_x[i] for i in range(sample_size)], [1 for i in range(sample_size)], color='red', marker='.', alpha=0.8)
# ax.set_aspect(1)

K=[2,4,7,11,16]
lr=[0.01,0.02]
k=K[0]
center_list=Kmeans(k,input_x,sample_size)

# ax.scatter([center_list[i] for i in range(k)], [1 for i in range(k)], color='black', marker='o', alpha=0.8)
# ax.set_ylim([0, 2])
# plt.show()
# Calculate Gaussian Widths (different clusters assume  same Gaussian width)
dmax=max_distance(center_list,k)
delta=dmax/(np.sqrt(2*k))
print(delta)

# w_hi2op=np.array([1 for i in range(k)]).reshape(k,1)
# b_op=1
np.random.seed(42)
w_hi2op = np.random.rand(k,1)*2-1
b_op = np.random.rand()*2-1

for index_epoch in range(100):
    cost_total = 0
    dw_hi2op = np.array([0.0 for i in range(k)]).reshape(k,1)
    db_op=0
    for index_sample in range(sample_size):
        x=input_x[index_sample]
        exp_output=output_h[index_sample]
        # def BackProp_op2hi(input_x, xj_list,w_hi2op, b_op, exp_output, act_output,k):
        act_output = Forward(x, center_list,delta, w_hi2op, b_op,k)
        (dw_hi2op,db_op)=BackProp_op2hi(x,center_list,w_hi2op,b_op,exp_output,act_output,k)
          
        (w_hi2op, b_op, dw_hi2op, db_op) = Weight_update(w_hi2op, b_op, dw_hi2op, db_op,gama=0.01)
    for index_sample in range(sample_size):
        x=input_x[index_sample]
        exp_output=output_h[index_sample]
        act_output = Forward(x, center_list,delta, w_hi2op, b_op,k)
        cost_total += CostFunction(act_output, exp_output)
    
    if index_epoch % 10 == 0:
            print('Iteration %s: Total Loss is %s' % (iter, cost_total))
    
print(w_hi2op)
print(b_op)

# Plot t
input_x_final=np.arange(0.0,1 ,0.00001)
final_output=[]
for index_sample in range(100000):
        x=input_x_final[index_sample]
        act_output = Forward(x, center_list,delta, w_hi2op, b_op,k)
        final_output.append(act_output)
ax = plt.gca()
ax.scatter(list(input_x_final),final_output , color='red', marker='.', alpha=0.8)
ax.scatter(list(input_x),list(output_h) , color='blue', marker='.', alpha=0.8)

x_list=np.arange(0.0,1 ,0.0001)
y_list=0.4*np.sin(2*np.pi*x_list)+0.5
ax.scatter(list(x_list),list(y_list) , color='black', marker='o', alpha=0.8)
# ax.set_aspect(1)
ax.set_ylim([0, 1])
plt.show()