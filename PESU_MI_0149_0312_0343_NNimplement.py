'''
Design of a Neural Network from scratch

*************<IMP>*************

a) Number of Layers: 2 Hidden Layers, 1 I/P Layer, 1 O/P Layer

b) Number of Neurons per Layer:
    I/P layer: We have 11 features because we one hot encoded the Community Feature
    1st Hidden Layer: 15 Neurons 
    2nd Hidden Layer: 15 Neurons
    O/P Layer: 1 Neuron as we are doing Binary Classifier
    
c) Dimensions of Weights and Bias Matrices:
    I/P Layer: X has dimensions 11x67  (As training split is 70%)
    1st Hidden Layer: 
        W1 = 15x11 (This is used in this implementation) 
        B1:  15x1 (For our implementation)
    2nd Hidden Layer: 
        W2 = 15x15 (This is used in this implementation) 
        B2:  15x1 (For our implementation)
    O/P Layer:
        W3 = 1x15 (This is used in our implementation)
        B3 = 1x1 (This is used in our implementation)

d) Activation functions:
    1st Hidden Layer: RELU Activation Function
    2nd Hidden Layer: RELU Activation Function
    O/P Layer: Sigmoid, as we are doing Binary Classification

e) Loss Function used:
    We used the Binary Cross Entropy Loss Function
    Loss = - Sigma(1 to m) [ y*log(1-a) + (1-y)*log(1-a)]

f) Additional Components used:
    1. We implemented Batch Gradient Descent, which basically takes input data in form of batches, 
    which helps to avoid the local minima
    
    2. We took advantge of Stratified Split, which distributes our result feature
    in an almost equal distribution between the training and test splits, which help
    the model to better generalise on new data
    

Mention hyperparameters used and describe functionality in detail in this space

'''
#Libraries Used

import numpy as np
import pandas as pd
import random
# SKLEARN IS ONLY USED FOR TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
#np.random.seed(1)
#Importing Cleaned Dataset
ds = pd.read_csv('cleaned_LBW_Dataset.csv')

#One Hot Encoding for Community Categorical Feature
cm = ds['Community']
new_cm = pd.get_dummies(cm, drop_first = True, prefix = 'Commun')
ds_new = pd.concat([ds,new_cm],axis = 1)
ds_new.drop(columns = ['Community'], inplace = True)

# Predictors and Response
X = ds_new.drop(columns = ['Result'])
y = ds_new['Result']

#Splitting into Train and Test Splits
#Do Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, stratify = y)#, random_state = 25)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns = list(X_train))
X_test = pd.DataFrame(scaler.transform(X_test), columns = list(X_test))
#print(y_train)
#y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)
#print(X_train)
#NN DESIGN

#Computing the Z value of Neuron: Example: WX + b
def compute_Z(weights, A, b):
    return (np.dot(weights,A) + b)

#Computing Activation Neuron
def compute_Act(Z, activ):
    return activ(Z)

#Loss and Activation Functions

#Sigmoid Function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

#Linear Function
def lin(z):
    return z

#Log Loss Function
def binary_crossentr(A,Y):
    return -np.sum(Y*np.log(A) + (1 - Y)*np.log(1 - A))*(1/(Y.shape[1])) #+ (lbd/(2*Y.shape[1]))*()

#Relu function
def relu(Z):
    '''
    Another Method
    f = Z>0
    Z = f*Z
    return Z
    '''
    return np.maximum(0,Z)

#Tanh Function    
def tanh(z):
    return ( (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)) )

#Derivatives of Activations

#Relu Derivative
def relu_derv(Z):
    t = Z>0
    return t*1

#Tanh Derivative
def tanh_derv(Z):
    return 1 - (tanh(Z)**2)

#Linear Derivative
def lin_derv(Z):
    return 1

#Sigmoid Derivative
def sigmoid_derv(Z):
    return sigmoid(Z)*(1-sigmoid(Z))

#X Shape is Nx x M
#O/P is 1 neuron -> Binary Classification

#Neural Network Functions

#NN Pipeline - Layers with Activations are added in this
NN = []

#To add a Layer with Activation
def add_layer(Nh, activ):
    NN.append([Nh,activ])
    
    
#Randomly Initialisation Weights  
def get_weights_initial(NN,X):
    l = len(NN) #No of Hidden Layers
    #print(l)
    # No of Weight Matrices are l+1
    Ws = {}
    #Ws['W1'] = np.random.randn(NN[0][0],X.shape[0])*0.01
    Ws['W1'] = np.random.randn(NN[0][0],X.shape[0])*(2/(X.shape[0]))
    #Ws['W'+str(l+1)] = np.random.randn(1,NN[-1][0])*0.01
    Ws['W'+str(l+1)] = np.random.randn(1,NN[-1][0])*(2/(NN[-1][0]))
    for i in range(2,l+1):
        Ws['W'+str(i)] = np.random.randn(NN[i-1][0], NN[i-2][0])*(2/(NN[i-2][0]))#*0.01
        
    return Ws

#Initialising all Biases as Zeros
def get_bias_initial(NN):
    l = len(NN)
    Bs = {}
    for i in range(1,l+1):
        Bs['b'+str(i)] = np.zeros([NN[i-1][0],1])
    Bs['b'+str(l+1)] = np.zeros([1,1])
    return Bs

#Computing Forward Pass through NN
def forward_pass(NN, Ws, Bs,X):
    Zs = {}
    As = {}
    l = len(NN)
    Zs['Z1'] = compute_Z(Ws['W1'],X,Bs['b1'])
    As['A1'] = compute_Act(Zs['Z1'],NN[0][1])
    for i in range(2, l+1):
        Zs['Z'+str(i)] = compute_Z(Ws['W'+str(i)], As['A'+str(i-1)], Bs['b' + str(i)] )
        #print(i)
        #print(Zs['Z'+str(i)].shape)
        As['A'+str(i)] = compute_Act(Zs['Z'+str(i)],NN[i-1][1])
    Zs['Z'+str(l+1)] = compute_Z(Ws['W'+str(l+1)], As['A'+str(l)], Bs['b' + str(l+1)] )
    As['A'+str(l+1)] = compute_Act(Zs['Z'+str(l+1)],sigmoid)
    return Zs, As #Returning Z's and Activation O/Ps of Neurons


# Back Propogation in Last Layer (Since Binary Classification: Sigmoid Layer)
def back_lastlayer(A_l,A_l2,Y, NN, lbd, Ws):
    dZ = (A_l - Y)*(1/Y.shape[1])
    dW = np.dot(dZ,A_l2.T) + (lbd/(Y.shape[1]))*Ws['W' + str(len(NN) + 1)]
    db = (np.sum(dZ, axis=1, keepdims=True))
    
    return dZ, dW, db #Returning Gradient for Last Layer

# Implementation of Back Propogation

def backprop(NN,Zs,As,Ws,Bs,Y,X,lbd):
    #Initialising Gradient Dictionaries
    
    dA = {}
    dZ = {}
    dW = {}
    db = {}
    
    #For Last Layer
    l = len(NN)
    dZ['Z'+str(l+1)], dW['W'+str(l+1)], db['b'+str(l+1)] = back_lastlayer(As['A'+str(l+1)], As['A'+str(l)], Y, NN, lbd, Ws)
    
    m = (1/Y.shape[1])
    for i in range(l, 1, -1):
        dA['A'+str(i)] = np.dot(Ws['W'+str(i+1)].T,dZ['Z'+str(i+1)])
        if(NN[i-1][1]==relu):
            gdash = relu_derv
        if(NN[i-1][1]==tanh):
            gdash = tanh_derv
        if(NN[i-1][1]==sigmoid):
            gdash = sigmoid_derv
        if(NN[i-1][1]==lin):
            gdash = lin_derv
        
        dZ['Z'+str(i)] = (dA['A'+str(i)] * gdash(Zs['Z'+str(i)]))
        dW['W'+str(i)] = np.dot(dZ['Z'+str(i)], As['A'+str(i-1)].T) + (lbd/(Y.shape[1]))*Ws['W' + str(i)]
        db['b'+str(i)] = np.sum(dZ['Z'+str(i)], axis = 1, keepdims = True)
        
    dA['A1'] = np.dot(Ws['W2'].T,dZ['Z2'])
    if(NN[0][1]==relu):
        gdash = relu_derv
    if(NN[0][1]==tanh):
        gdash = tanh_derv
    if(NN[0][1]==sigmoid):
        gdash = sigmoid_derv
    if(NN[0][1]==lin):
        gdash = lin_derv

    dZ['Z1'] = (dA['A1'] * gdash(Zs['Z1']))
    dW['W1'] = np.dot(dZ['Z1'], X.T) + (lbd/(Y.shape[1]))*Ws['W1']
    db['b1'] = np.sum(dZ['Z1'], axis = 1, keepdims = True)
    
    return dW,db,dZ,dA #Returns Dictionaries of all Gradients

#Function to Update Weights & Biases        
def update_weights(Ws,Bs,dW,db,lr,NN):
    l = len(NN)
    for i in range(1,l+2):
        Ws['W'+str(i)] = Ws['W'+str(i)] - (lr*dW['W'+str(i)])
        Bs['b'+str(i)] = Bs['b'+str(i)] - (lr*db['b'+str(i)])
    return Ws,Bs #Returns Updated Weights & Biases

#Gets Predictions from Final Activation Values
def get_yhat(As,l):
    t = As['A'+str(l+1)] >= 0.6
    t = t*1
    return t
    
# Calculating Accuracy
def Accur(yhat,yts):
    arr = yhat - yts
    #print(arr)
    pos = np.count_nonzero(arr==0)
    #print(pos)
    
    return (pos/yts.shape[1])*100

#Vanilla Implementation of Gradient Descent
def gradient_descent(NN,X,Y,itr,lr):
    l = len(NN)
    Ws = get_weights_initial(NN,X)
    Bs = get_bias_initial(NN)
    
    for epoch in range(itr):
        
        X = X.sample(frac = 1)
        
        
        new_Zs, new_As = forward_pass(NN,Ws,Bs,X)
        #print(new_Zs)
        #print(new_As)
        #if(epoch%50==0):
           # print('epoch',epoch+1)
           # print('Cost:',binary_crossentr(new_As['A'+str(l+1)],Y))
            
        dW,db,dZ,dA = backprop(NN,new_Zs,new_As,Ws,Bs,Y,X)
        Ws, Bs = update_weights(Ws,Bs,dW,db,lr,NN)
        
    new_Zs, new_As = forward_pass(NN,Ws,Bs,X)    
    print('Final Cost:',binary_crossentr(new_As['A'+str(l+1)],Y))
    ytr_hat = get_yhat(new_As,l)
    print('Train Accuracy is:',Accur(ytr_hat,Y))
    return Ws, Bs #Returning Weights and Biases

# Implementation of Batch Gradient Descent
def Batch_GD(NN,X,Y,itr,lr,bat_siz,lbd):
    l = len(NN)
    Ws = get_weights_initial(NN,X)
    Bs = get_bias_initial(NN)
    
    for epoch in range(itr):
        
        new_xy = (pd.merge(X.T, Y.T, left_index=True, right_index = True)).sample(frac=1)
        X = new_xy.drop(columns = ['Result']).T
        Y = new_xy[['Result']].T
        #print(type(Y))
        
        for batch in range(0,X.shape[1],bat_siz):
            
            X_batch = X.iloc[:,batch:(batch+10)]
            Y_batch = Y.iloc[:,batch:(batch+10)]
            Y_batch = Y_batch.values
            
        
            new_Zs, new_As = forward_pass(NN,Ws,Bs,X_batch)
            #print(new_Zs)
            #print(new_As)
            #if(epoch%50==0):
                #print('epoch',epoch+1)
                #print('Batch Cost:',binary_crossentr(new_As['A'+str(l+1)],Y_batch))

            dW,db,dZ,dA = backprop(NN,new_Zs,new_As,Ws,Bs,Y_batch,X_batch,lbd)
            Ws, Bs = update_weights(Ws,Bs,dW,db,lr,NN)
        
    new_Zs, new_As = forward_pass(NN,Ws,Bs,X)  
    print('---------------RESULTS-----------------')
    #print('Final Binary Cross Entropy Training Cost:',binary_crossentr(new_As['A'+str(l+1)],Y))
    ytr_hat = get_yhat(new_As,l)
    print('Train Accuracy is:',Accur(ytr_hat,Y))
    print('')
    return Ws, Bs # Returning Weights and Biases
    


# Final Model Selected

#Adding Layers with Activations
NN = []
add_layer(5000, relu)
#(25,relu)
#add_layer(10,relu)
#add_layer(10,relu)
#add_layer(10,relu)
#add_layer(5,relu)
add_layer(2,relu)

#Initially we trained with Vanilla Gradient Descent
#To improve our Model, we implemented Batch Gradient Descent

#Final_Ws, Final_Bs = gradient_descent(NN,X_train.T,y_train.T,1000,0.01)
Final_Ws, Final_Bs = Batch_GD(NN,X_train.T,y_train.T,500,0.1,5,0.5)

#On Test Data
new_Zs, new_As = forward_pass(NN,Final_Ws,Final_Bs,X_test.T)
l = len(NN)
#print(new_As)
print('Binary Cross Entropy Test Cost:',binary_crossentr(new_As['A'+str(l+1)],y_test.T))
yhat = get_yhat(new_As,l)
print('Test Accuracy is:',Accur(yhat,y_test.T))
print('')

#Function to compute Confusion Matrix

def CM(y_test,y_test_obs):
		'''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''
		
		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0
		
		for i in range(y_test_obs.shape[1]):
			if(y_test[0][i]==1 and y_test_obs[0][i]==1):
				tp=tp+1
			if(y_test[0][i]==0 and y_test_obs[0][i]==0):
				tn=tn+1
			if(y_test[0][i]==1 and y_test_obs[0][i]==0):
				fp=fp+1
			if(y_test[0][i]==0 and y_test_obs[0][i]==1):
				fn=fn+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)
		
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")
			
CM(y_test.T,yhat)











