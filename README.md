# MSCS335-HW9
-------HW2-------

Dataset: 
This dataset comes from kaggle with a multitude of different statistics about global energy. There is a total of 129 quantitative variables and 300 individuals 

Results: ![image](https://github.com/user-attachments/assets/0b639edf-3068-4082-ab23-f08d9af3d898)
When using a test size of around 0.3 it is apparent that overfitting is occuring as the linear model is getting almost 100% accuracy although with very large coefficents. 

If the test size is upped to about 0.85 ![image](https://github.com/user-attachments/assets/068095c4-ca6b-4918-a038-0b210df8ecdb)
Then we see less overfitting as the dataset being trained is much smaller part of the data. Ridge is still performing better overall with R^2 and RMSE values typically better than linear although I presume that this data is very linear as there is not much difference with linear even beating ridge in some cases.


------HW3--------

Dataset: 
Utilizes both the previous dataset and a dataset about Tree Survival. In this new dataset there we are using Predicting if tree is alive 1=dead or  0=alive based on these:
Light_ISF=Light level reaching each subplot
AMF=percent of arbuscular mycorrhizal fungi in root of havested seeds
Phenolics=Gallic acid equivalents(nmol) per mg of dry extract
NSC=Percent of dry mass nonstructured carbohydrates
There is a total of about 7770 datapoints in the dataset

Confusion Matrix:
![Tree_Survival](https://github.com/user-attachments/assets/aa182a72-892e-4324-9d7b-f97188bc1d3a)

Results:
Using a SVC and a grid search for the best C value the model was able to correctly predict if a tree was to be alive 663 times with 105 false negatives. And correctly predict if the tree was dead 40 times with 27 false positives. The model has an overall accuracy of 0.842 but this can be higher if all trees are predicted as alive. We do want to predict if a tree is dead of not because that has a great deal of information that can be taken away so an overall decrease for an increase in that accuracy is ideal.


----------HW7-----------

Dataset:
This dataset is a wine classification dataset that aims to classify wine based on Alcohol, Malic_Acid, Ash, Ash_Alcanity, Magnesium, Total_Phenols, Flavanoids, Nonflavanoid_Phenols, Proanthocyanins, Color_Intensity, Hue, OD280, and Proline. 
There is a total of 178 datapoints.

Elbow Graph:
![Elbow_Graph](https://github.com/user-attachments/assets/f0a731fa-ac83-4d6a-afee-381884b668c3)

Results:
Using the elbow graph method to find the appropriate number of clusters. I deciced to compare 3 and 4 clusters for what their different results would be. Utilizing this the inertia of 3 clusters is 1187.419643637654 while the inertia for 4 clusters is 1098.1934158908357. While this is a little bit better for 4 there isn't really a right answer. In the description of the dataset these wines come from 3 overall types of wine but there can be other datapoints that are more meaningful to split into more clusters.

--------HW8------------

Dataset:
This dataset utilizes sales data from differen mediums in the United States. There is a total of 17992 rows and I decided to utilize the columns Order Quantity: Quantity of products ordered, Discount Applied: Applied discount for the order, Unit Cost: Cost of a single unit of the product, and Unit Price: Price at which the product was sold. These were the only quantitative variables. In this neural network I aimed to do a regression to predict minimize the MSE in the price of the product. 

Neural Network Architecture:
ReLu
1-8
8-4
4-1
epochs=100, batch_size=16, lr=0.001
Train on 50 epochs

Results:
MSE of 0.00097

This MSE is a very low number and I presume that there is overfitting happening as there is a sharp drop off in what the loss at about 50 epochs of training. The NN is fed a lot of data that is all very correlated to each other and even with normalization there is still times where a specific datapoint can be singled out.

