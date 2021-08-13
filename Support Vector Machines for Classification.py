#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines for Classification

# # 1. Linear Support Vector Machine
# The first thing we're going to do is look at a simple 2-dimensional data set and see how a linear Support vector machine works on the data set for varying values of C (similar to the regularization term in linear/logistic regression). Let's load the data.

# In[1]:


import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from scipy.io import loadmat  
get_ipython().run_line_magic('matplotlib', 'inline')

# load data
raw_data = loadmat('ex6data1.mat')
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])  
data['y'] = raw_data['y']
X = np.array(data[['X1', 'X2']])
y = np.array(data['y'])

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)


# ## Implementation
# Notice that there is one outlier positive example that sits apart from the others. The classes are still linearly separable but it's a very tight fit. We're going to train a linear support vector machine to learn the class boundary. Use scikit-learn SVC to linearly classify the dataset. In svm the regularization is controled by the parameter C. Informally, the C parameter is a positive value that controls the penalty for misclassified training examples. A large C parameter
# tells the SVM to try to classify all the examples correctly. C plays a role similar to $\frac{1}{\lambda}$ , where $\lambda$ is the regularization parameter that we were using previously for logistic regression. Classify the training examples with C=1 and C=100. Plot the decision boundary (or the confidence level (distance from boundary)). Explain what you observe.

# In[2]:


from sklearn import svm 
#TODO: use SVC to linearly classify
#use kernel='linear', and C=[1,100] in call to SVC
svclassifier = svm.SVC(kernel = 'linear', C=1)


# In[3]:


svclassifier.fit(X, y)


# ## Visualization
# plot support vectors

# ### Visualization at C = 1

# In[4]:


svclassifier = svm.SVC(kernel = 'linear', C=1)
svclassifier.fit(X, y)


# In[5]:


# Some helper functions

# Set min and max values and give it some padding
x_min, x_max = X[:,0].min() - .1, X[:,0].max() + .1
y_min, y_max = X[:,1].min() - .1, X[:,1].max() + .1
h = 0.01

# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
d = np.c_[xx.ravel(), yy.ravel()]
    
# Predict the function value for the whole grid
Z = svclassifier.predict(d)
Z = Z.reshape(xx.shape)
    
# Plot the contour and training examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=.5)
    
# plot the postive and negative planes
G = svclassifier.decision_function(d)
G = G.reshape(xx.shape)
plt.contour(xx, yy, G, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
# plot support vectors
# TODO: plot the support vectors using "support_vectors_" attribute of SVC


        
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
            
plt.title('Decision Boundary')
plt.show()


# ### Visualization at C = 25

# In[6]:


svclassifier = svm.SVC(kernel = 'linear', C=25)
svclassifier.fit(X, y)


# In[7]:


# Some helper functions

# Set min and max values and give it some padding
x_min, x_max = X[:,0].min() - .1, X[:,0].max() + .1
y_min, y_max = X[:,1].min() - .1, X[:,1].max() + .1
h = 0.01

# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
d = np.c_[xx.ravel(), yy.ravel()]
    
# Predict the function value for the whole grid
Z = svclassifier.predict(d)
Z = Z.reshape(xx.shape)
    
# Plot the contour and training examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=.5)
    
# plot the postive and negative planes
G = svclassifier.decision_function(d)
G = G.reshape(xx.shape)
plt.contour(xx, yy, G, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
# plot support vectors
# TODO: plot the support vectors using "support_vectors_" attribute of SVC


        
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
            
plt.title('Decision Boundary')
plt.show()


# ### Visualization at C = 50

# In[8]:


svclassifier = svm.SVC(kernel = 'linear', C=50)
svclassifier.fit(X, y)


# In[9]:


# Some helper functions

# Set min and max values and give it some padding
x_min, x_max = X[:,0].min() - .1, X[:,0].max() + .1
y_min, y_max = X[:,1].min() - .1, X[:,1].max() + .1
h = 0.01

# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
d = np.c_[xx.ravel(), yy.ravel()]
    
# Predict the function value for the whole grid
Z = svclassifier.predict(d)
Z = Z.reshape(xx.shape)
    
# Plot the contour and training examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=.5)
    
# plot the postive and negative planes
G = svclassifier.decision_function(d)
G = G.reshape(xx.shape)
plt.contour(xx, yy, G, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
# plot support vectors
# TODO: plot the support vectors using "support_vectors_" attribute of SVC


        
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
            
plt.title('Decision Boundary')
plt.show()


# ### Visualization at C = 75

# In[10]:


svclassifier = svm.SVC(kernel = 'linear', C=75)
svclassifier.fit(X, y)


# In[11]:


# Some helper functions

# Set min and max values and give it some padding
x_min, x_max = X[:,0].min() - .1, X[:,0].max() + .1
y_min, y_max = X[:,1].min() - .1, X[:,1].max() + .1
h = 0.01

# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
d = np.c_[xx.ravel(), yy.ravel()]
    
# Predict the function value for the whole grid
Z = svclassifier.predict(d)
Z = Z.reshape(xx.shape)
    
# Plot the contour and training examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=.5)
    
# plot the postive and negative planes
G = svclassifier.decision_function(d)
G = G.reshape(xx.shape)
plt.contour(xx, yy, G, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
# plot support vectors
# TODO: plot the support vectors using "support_vectors_" attribute of SVC


        
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
            
plt.title('Decision Boundary')
plt.show()


# ### Visualization at C = 100

# In[12]:


svclassifier = svm.SVC(kernel = 'linear', C=100)
svclassifier.fit(X, y)


# In[13]:


# Some helper functions

# Set min and max values and give it some padding
x_min, x_max = X[:,0].min() - .1, X[:,0].max() + .1
y_min, y_max = X[:,1].min() - .1, X[:,1].max() + .1
h = 0.01

# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
d = np.c_[xx.ravel(), yy.ravel()]
    
# Predict the function value for the whole grid
Z = svclassifier.predict(d)
Z = Z.reshape(xx.shape)
    
# Plot the contour and training examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=.5)
    
# plot the postive and negative planes
G = svclassifier.decision_function(d)
G = G.reshape(xx.shape)
plt.contour(xx, yy, G, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
# plot support vectors
# TODO: plot the support vectors using "support_vectors_" attribute of SVC


        
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
            
plt.title('Decision Boundary')
plt.show()


# ## Answer the following Questions
#     1 What effect has the parameter C on the decision boundary?
#     2 What are support vectors, why do support vectors change by changing C?
#     3 Use the decision_function of SVC to find the margin of the farthest positive and farthest negative data point from the decision boundary
#     

# **1. Effect of Parameter C on Decision Boundary**
# 
# `C` is the regularization parameter, its value determines the margin hyperplane. The use of this parameter is to avoid misclassifying any data points. As shown in the different training instances with increasing values of C, it is clear that the `higher values of C` will narrow down the margin hyperplane and contrary to that, the `smaller the value of C` gets, the broader (larger) the margin hyperplane gets and for very small values of there will be considerable misclassification of datapoints.
# 
# **2. Support Vectors and why do they change with `C`**
# 
# Support Vectors are data points which are comapratively closer to the margin hyperplane. These points are called Support Vectors because they are used in maximizing the margin of the classifier. 
# 
# Changing the value of `C` changes the margin hyperplane, and thus the points closer to the hyperplane also change and thus become the new support vectors. Moreover, if the previous support vectors are deleted, it will also change the support vector to some new point.
# 

# **3. Margin of Farthest Positive and Farthest Negative Data Points from Decision Boundary**

# In[14]:


svclassifier = svm.SVC(kernel = 'linear')


# In[19]:


svclassifier.fit(X,y)


# Farthest positive

# In[21]:


max(svclassifier.decision_function(X))


# Farthest Negative

# In[22]:


min(svclassifier.decision_function(X))

