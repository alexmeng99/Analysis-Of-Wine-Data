*********
## Analysis of Wine Data
*********

Abner Espinoza
Edward Wu
Alexandria Meng

# Abstract

It’s no secret that everything from sugar to acidity affects the taste and quality of food, but namely wine. With the given dataset, we set out to determine if we could predict both the quality of the given wines and the type of wine (red or white) using various machine learning models. This project provides a graphical and statistical analysis of the Wine Quality Data Set provided to us by the UCI Machine Learning Repository. 

## Data Exploration
# 1.1 Understanding the Data

First, we decided to look into what problems we could solve based on what the data looks like. Since the data is tabular, we brainstormed that Decision Trees could yield positive results and point us in the right direction. As we’re primarily interested with how close our predictions are to the actual values, we further presumed that an error metric such as Mean Squared Error (MSE) would best help us assess the accuracy of our models and allow us to compare them to one other. We also used Standard Error as an overarching benchmark. Lastly, we shuffled the data and split it into training and testing data at a 75:25 rate.

# 1.2 Visualizing the Data

To grant a better understanding of the Wine Data, we plotted multiple scatter plots and histograms (not all shown below) in order to examine how the features compared against Quality, as well as how many data points were in each feature. This allowed us to see each feature’s range.
   





<img width="=1074" alt="Screen Shot 2021-12-10 at 2 54 00 PM" src="https://cdn.discordapp.com/attachments/766599919813263361/955905901846470736/unknown.png">



    Figure 1: Quality compared to Fixed Acidity (Red)		 Figure 2: Quality compared to Fixed Acidity (White)
Next, we compared features to one another to detect if there were patterns or relationships we could pick up by eye. Namely, we were looking for positive, linear relationships. This was in the hopes that these features would provide more insight as we performed model exploration. We chose two combinations that we found to have these types of relationships: Alcohol & pH, and Chlorides & Density. As a result of this, we believed that using Dimensionality Reduction with these features would be the right place to start. With two models at hand (Decision Trees and Dimensionality Reduction) we concluded this aspect of data exploration and moved onto model exploration, which itself would help us to further understand the data.

## Model Exploration
# 2.1 Dimensionality Reduction

Using the aforementioned features (Alcohol & pH, and Chlorides & Density), we used Dimensionality Reduction to determine if wine quality could be predicted. There was only one parameter that we considered:
- n_components- desired dimensions in the output of the transform.

When using the Alcohol & pH features against Quality, an average Standard Error of 43% was calculated using the Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) models. Both models had their n_components parameters set to 1 (since there are only two features), and both MSEs were recorded to be around 0.96. With the Density & Chlorides features, the average Standard Error was 41% for both PCA and LDA, with their n_components set to 1, and with an MSE of about 1.05 for both models. This showed us that linear relationships between features didn’t accurately map to Quality.

Taking into consideration all the features of the data (barring Quality), we found that setting the n_components to 5 gave us 41.5% Standard Error using the PCA method, with our highest being 44% using LDA. However, the MSEs were all around 0.94, which was too high for comfort. 

This means that, on average, our Quality predictions were nearly a full point off of the actual Quality value. However, this showed us that using all the features was an improvement from focusing on just the linearly correlated features. As a result, we changed our processes to include all of the features rather than isolating them. For this dataset, we quickly found that taking into account more features yields better results.

## 2.2 Random Forest Classifier (RFC)

Before we experimented with RFCs, we ran a simple Decision Tree on our data. Using mltools, we iterated from 1-100 for both the minParent and minLeaf parameters and from 1-15 for the maxDepth parameter. For the red wine, we were able to achieve an MSE of 0.5094 by setting minParent to 1, minLeaf to 6, and maxDepth to 6. For the white wine, we achieved an MSE of 0.6217 by setting minParent to 2, minLeaf to 7, and maxDepth to 5.  

We decided that taking an ensemble method that involves bagging would work well given our optimistic findings using the Decision Trees. In short, RFCs create a number of trees and the ultimate prediction is decided by committee. Therefore, it is impossible to memorize the training data and this prevents overfitting. For this model, we emphasized exploration of the Red Wine data.

Using Scikit Learn’s Random Forest Classifier ensemble package, there were a few parameters to consider:
- n_estimators- number of trees in the forest.
- Criterion- “gini” or “entropy.”
- Max_depth- maximum depth in a tree.
- Min_samples_split- The minimum number of samples required to split an internal node.
- Min_samples_leaf- The minimum number of samples required to be at a leaf node.
	
To determine the hyperparameters that would produce the most accurate results, we varied the values in an attempt to minimize the MSE. We used similar ranges to what we’ve previously experienced working with.

<img width="=931" alt="Screen Shot 2021-12-10 at 2 54 00 PM" src="https://cdn.discordapp.com/attachments/766599919813263361/955907106589012008/unknown.png">


          Figure 1: varying the number of 	                           Figure 2: varying the maximum 
     		    estimators for 100-1200.  		                             depth  for 1-30.

Using these MSE values, we determined the hyperparameter values that most optimize the performance of the RFC. Setting criterion to “entropy”, max_samples to 750, n_estimators to 300, max_depth to 15, min_samples_split to 5, and min_samples_leaf to 2, we managed an MSE of 0.3969. Additionally, using GridSearchCV (which implements cross-validation over a grid of parameters), we were able to determine an even more optimized result by setting max_samples to None, n_estimators to 300, max_depth to 15, min_samples_split to 2, min_samples_leaf to 1, ultimately achieving an MSE of 0.3531.

Using the same parameters, we also trained a model to predict whether a certain wine was red or white based on the features. Our RFC achieved a Standard Error of 99.54% in predicting between red and white wine.

## 2.3 Neural Network

Next, we used Scikit Learn’s neural network package for their Multi-Layer Perceptron model. There were three hyperparameters to consider:
- solver- function for weight optimization.
- hidden_layer_sizes- the number of hidden layers and neurons per layer.
- alpha- L2 penalty.

By again using GridSearchCV, we were able to refine and narrow the parameters of the network down to
For the red wine — setting solver to ‘adam’ (a stochastic gradient-based optimizer), alpha to 0.00001, and hidden_layers to 10, 4.
For the white wine — setting solve to ‘adam’, alpha to 0.00001, and hidden_layers to 11, 5.

Using these parameters, we were able to achieve an MSE of 0.7775 for the red wine and 0.7625 for the white wine. To further prospect the data, we decided to use the mltools package alongside Scikit Learn. We varied the number of hidden layers from 5 to 30 and the number of nodes from 1 to 64. Using this method, we were able to achieve an MSE of 0.915 for the red wine. After comparing the results for the red wine between Scikit Learn and mltools, we decided the latter method was not worth pursuing further.

## 2.3 Support Vector Machine (SVM)

For our last model, we realized that using Support Vector Machines for classification would be effective, especially if we could use Kernel tricks to map the features to a higher dimension. Using the Radial Basis function kernel, we were able to minimize the MSE to 0.4531. Other kernel functions such as linear, polynomial, and sigmoid yielded 0.5063, 0.4688, 0.625 respectively. By varying the hyperparameters, we managed to improve our model and lower the MSE to 0.45. We also utilized the confusion matrix and classification report from Scikit Learn to further analyze the accuracy of our predictions.

From that, there were a few approaches we considered to improve the accuracy. Similar to the RFC, we ran a Grid Search to determine the optimal hyperparameters but the MSE did not improve. We decided SVMs were not worth pursuing further. However, we acknowledge that implementing regularization, removing features, using the libsvm library, and determining a better error metric than MSE such as weighted accuracy has the potential to improve our model.
Conclusion
In this project, we started with the assumption that patterns visualized through plots would point us to the most optimized machine learning models, and that wine quality could be predicted with a few linearly correlated features. Through trial and error, we concluded that the data we were working with was much more complex. Wine quality could be modestly predicted when taking into consideration many features with Dimensionality Reduction, Random Forest Classifiers, Neural Networks, and Support Vector Machines. As RFCs yielded the smallest MSE between all of the models, we determined it to be the best model for predicting wine quality (Mean Squared Error ~0.3531). With further exploration, we discovered wine type (red or white) could be predicted at an extremely high accuracy also using RFCs (Standard Error ~99%). 

The discrepancy between the accuracy of predicting the wine quality (for red and white wine) and predicting the type of wine can be explained by the presence of certain qualities exclusive to red wine and others being exclusive to white wine. While both wines shared the same features, some features exclusively set the wines apart, making it easier for models to predict the type of wine. This contrasts predicting wine quality where, given the limited sizes of the individual wine data, the models can struggle to make precise predictions. We believe that this is the main reason our Neural Networks and Support Vector Machine models failed to have MSEs lower than 0.40, and on the other hand why our RFC was able to excel ahead of the rest of the models. We also presume that our Dimensionality Reduction models did not perform very well on our data due to it not having high enough input variables. 

For future improvements to our probing of this data, better error metrics when determining the quality of the wine such as weighted accuracy, and better utilizing the confusion matrix and the classification report should all be considered and scrutinized to best eliminate underfitting and overfitting in more advanced models. As we saw, by eliminating the possibility of memorizing the data using RFCs, machine learning models can be improved significantly and drastically decrease the MSE.

Dimensionality Reduction was handled by Alex, Random Forests and SVMs by Edward, and Decision Trees and Neural Networks by Abner. The data preprocessing was collaborated on equally, as was the data exploration. As we individually worked on the machine learning models and as a result individually learned more about the data, we met up often to discuss our findings and how we should proceed. We also helped improve one another’s models, which is how we managed to achieve such a low MSE for the RFC.







*********

      References
       [1] https://archive.ics.uci.edu/ml/datasets/Wine+Quality       
       [2] https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
       [3] https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6
       [4] https://towardsdatascience.com/which-machine-learning-model-to-use-db5fdf37f3dd
       [5] https://towardsdatascience.com/understanding-random-forest-58381e0602d2
       [6] https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769
       [7] https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/
       [8] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
       [9] https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
       [10] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
