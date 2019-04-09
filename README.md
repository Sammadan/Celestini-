# Celestini-Q4
Support Vector Machine” (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. However,  it is mostly used in classification problems.    
In this we have to find a zoo for every animal given a set of features for the animal.   
After downloading the dataset we converted output classes to categorical variables using One Hot Encoding. 
SVM kernel is used to take data as an input and transform it into the required form. Different SVM algorithms use different types of kernels.
    The kernels used are :
Linear Kernel                  
Radial Basis Kernel
Polynomial Kernel
Sigmoid Kernel

      1. Linear Kernel                                                            Accuracy:        1.0
    It is useful when dealing with large sparse data vectors. It is often used in text categorization. The splines kernel also              performs well in regression problems. 
   
      2. Polynomial Kernel                                           Accuracy:      0.9
    It is popular in image processing.
          
      3. Radial Basis Kernel                                                    Accuracy:         0.76
    It is general-purpose kernel; used when there is no prior knowledge about the data
          
      4. Sigmoid Kernel                                                 Accuracy:        0.75
    We can use it as the proxy for neural networks.   
           
# Neural Networks
Using keras, a MLP was designed to predict classes. OneHotEncoder was again used to convert labels to categories. 

Only one hidden layer was used with ReLU activation and 21 neurons. Softmax was used in the final layer as this was a classification problem. Softmax is used because it's output for each neuron depends on the output of every other neuron in the layer, making it ideal for classification problems.

Keras is used as it automatically performs back-propagation, without explicit coding. 

The model showed 100% accuracy on both train and test set, with 95% k cross validation score.
