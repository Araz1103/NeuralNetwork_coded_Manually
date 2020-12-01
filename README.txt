
TEAM NAME: PESU-MI_0149_0312_0343
Date: 17/11/2020
-------------------------------------------------- FULL DETAILS FOR PRE-PROCESSING-----------------------------------------------------
1. Attempt to counter NAN with traditional mean implementation column-wise:

 a) Initially, we thought to fill all Missing and NAN values with means of each column, independently. 
 b) However, we saw that because of outlier like behaviour, the values replaced were not accurate
 c) Had we took this approach, this could have led to a wrong bias in our data and inefficent results
 d) With this approach, the accuracy gotten will not be optimal 

2. Countering NAN Values with Data Imputation:
 a) Since the previous approach didn't pan out, we took an innovative detour to account for that error
 b) We observed, that the Community feature did not have any missing values
 c) Since it provided all data points, we decided to use this as grouping factor, to calculate missing values of other features
 d) For Example: For Community = 1, we took the mean, feature wise, of all non-missing values, and imputed the missing value in this group
    by that mean.
 e) This approach proved to be correct ,as now the mean & median are of a justified value, as they have been taken on a closed group.
 f) This leads to lesser variance as compared to traditional mean
 

 -------------------------------------------------- FULL DETAILS FOR IMPLEMENTATION-----------------------------------------------------
1. Explain your implementation:

a) Step 1 One Hot Encoding of Community Feature: As the Community Feature had 4 Classes: 1,2,3,4 , this can lead to a bias in data,
   as the difference between class 1 and class 4 is not the same as the difference between class 1 and 3.
   To solve this problem, we did One Hot Encoding, with help of the pandas: get_dummies function.
 
So now number of I/P Features becomes 11

b1) Step 2 Splitting the Dataset: We have used 70% Data for Training and 30% Data for Testing. Here, we have used an extra component, in 
   form of Stratified Splitting, to ensure an equal distribution of the response variable in both Train & Test splits
b2) Normalising the data with StandardScaler()

c) Step 3 Neural Design: We make functions for the following computations:
   1. Computing Z & Activations
   2. Computing Activation Functions and their Derivatives
   3. Computing Loss Function (In our case Binary Cross Entropy)
   4. Pipelining the NN 
   5. Adding Layers with Activations
   6. Initialising Weights and Biases
   7. Computing Forward Pass
   8. Computing Backpropogation - This is done in 2 parts. First we compute gradients for Last Layer, and then for subsequent previous Layers
   9. Updating the Weights and Biases
   10. Implementing Gradient Descent:
	A. We implemented the Standard Gradient Descent
        B. To improve our Model, we implemented Batch Gradient Descent
   11. Getting the Predictions
   12. Computing Accuracy
   13. Getting the Confusion Matrix

d) Experimenting to get the Optimal Neural Network Model:
	1. After testing all Activations, we found RELU to be relatively optimal
        2. After testing with 1 - 4 Layers, we found 2 layers to be the best
        3. After testing the Learning Rate, we found 0.01 to be optimal
        4. Testing for various Batch Sizes, to find the optimal
        5. We tested for iterations from 100 to 1000, and found 500 to be optimal

e) Computing Results: Accuracy, Precision, Recall and F1 Score

2. List your hyperparameters
  Number of Layers: 2
  Neruons per Layer: 15 in each
  Activation Function: RELU
  Number of Iterations: 500
  Learning Rate: 0.01
  Batch Size: 10
  np.random.seed: 1
  random_state (for split): 25



3. What is the key feature of your design that makes it stand out:
 A. Pipelined Structure of the Code: The entire code has been written in a manner, such that each function is like a lego block
    which can fit into the rest, to make bigger and better blocks. This helps to improve not only the code readibility, but also
    makes it easy to implement any new features like adam optimiser, momentum, batch normalisation, etc
 B. Mathematical Soundness of Code: We started this assignment by deriving the entire forward and back-propogation by hand, and getting 
    generalised formulas for each computation. Since we used that as a reference to code, all our functions are coded in a way to 
    represent Training and Learning of NNs mathematically. So to implement any new mathematical functions or techniques makes it easy. We
    also claim that any person with basic knowledge of calculus can understand our code :D

4. Have you implemented something beyond the basics:
 A. Batch Gradient Descent: Since we observed that our performance was low, we realised that the gradient descent is often getting stuck at
    local minima. To avoid that, we implemented Batch Gradient Descent, which is taking the data in batches, and then updating weights batch wise.
    After each Epoch, the data is shuffled and then again split into batches, as this helps to come out of local minimas, 
     and converge to a better value. We observed this to perform better than standard gradient descent.

 B. Stratified Split: Here, we have used an extra component, in form of Stratified Splitting, 
    to ensure an equal distribution of the response variable in both Train & Test splits

5. Detailed steps to run your files:
	Run the code in the same directory as of the source folder
	If running on Linux use python3 otherwise use python, and use dos2unix command if running in linux
  a) python PESU_MI_0149_0312_0343_preprocess.py		#This file generates Cleaned Dataset as a CSV file 
  b) python PESU_MI_0149_0312_0343_NNimplement.py              #This runs the Neural Network Implementation