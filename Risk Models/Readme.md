# Health Risk prediction models


# Project description:
- The objetive of this project is to predict the 10-year risk of death of individuals based on 18 different medical factors such as age, gender, systolic blood pressure, BMI etc.
- Two types of models were used : Linear model and random forest classifier model.
- Finally, we compare different methods for both models using c_index.

# Data exploration:
- For this project, I will be using the NHANES I epidemiology dataset.
- Looking at our training and validation data, we see that some of the data is missing: some values in the output of the previous cell are marked as NaN
("not a number"). Missing data is a common occurrence in data analysis, that can be due to a variety of reasons, such as measuring instrument malfunction, 
respondents not willing or not able to supply information, and errors in the data collection process.

![](assets/data.png)

- For each feature, represented as a column, values that are present are shown in black, and missing values are set in a light color.
From this plot, we can see that many values are missing for systolic blood pressure (Systolic BP).

### Imputation:
- Seeing that our data is not missing completely at random, we can handle the missing values by replacing them with substituted values based on the 
other values that we have. This is known as imputation.
- The first imputation strategy that we will use is **mean substitution**: we will replace the missing values for each feature with the mean of the available values. 
- Next, we will apply another imputation strategy, known as **multivariate feature imputation,** using scikit-learn's IterativeImputer class.
With this strategy, for each feature that is missing values, a regression model is trained to predict observed values based on all of the other features, 
and the missing values are inferred using this model. As a single iteration across all features may not be enough to impute all missing values, 
several iterations may be performed, hence the name of the class IterativeImputer.

# Libraries:


# Linear model:
- Linear regression is an appropriate analysis to use for predicting the risk value using multiple features. 
- It is used to find the best fitting model to describe the relationship between a set of features 
(also referred to as input, independent, predictor, or explanatory variables) and an outcome value
(also referred to as an output, dependent, or response variable).
- We need to transform our data so that the distributions are closer to standard normal distributions.First we will remove some of the skew from the 
distribution by using the log transformation. Then we will "standardize" the distribution so that it has a mean of zero and standard deviation of 1.
![](assets/process.png)


# Random forest classifier model:
- Random forests combine predictions from different decision trees to create a robust classifier.
![](assets/dt.png)
Decision tree classifier.
- The fundamental concept behind random forest is a simple but powerful one — the wisdom of crowds. 
In data science speak, the reason that the random forest model works so well is:
**A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models.**
- We need tune (or optimize) the hyperparameters, to find a model that both has good predictive performance and minimizes overfitting. The hyperparameters we choose to adjust will be:
  - **n_estimators:** the number of trees used in the forest.
  - **max_depth:** the maximum depth of each tree.
  - **min_samples_leaf:** the minimum number (if int) or proportion (if float) of samples in a leaf.

# Results:



