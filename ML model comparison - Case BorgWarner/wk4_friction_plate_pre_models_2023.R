## PRE-assignment file for Friction Plate prediction prepared by your Data-Science team

#LOAD NECESSARY LIBRARIES - install if needed
#install.packages("rpart"); install.packages("rpart.plot"); install.packages("randomForest"); install.packages("e1071"); install.packages("neuralnet"); install.packages("gbm"); install.packages("pROC")
library(rpart); library(rpart.plot)
library(randomForest)
library(e1071)
library(neuralnet)
library(gbm)
library(pROC)
library(caret)
library(ggplot2)
library(reprtree)


# READ DATA: First your task is to read the data and split it into training and test set.

# !set correct directory:
setwd("C:/Users/Tuomo/OneDrive - Aalto University/Creating Value with Analytics/Week 4 - Predictive analytics II/Case Week 4 Fast-tracking Friction Plate Validation with Predictive analytics-20230322")

friction_plate_data <- read.csv("friction_place_data_2023.csv", sep=",") #Read the data, note that it uses ',' (comma) as separator. The first column is number of observation, which is not needed and we drop it. 
head(friction_plate_data) #Check the data file and its first rows to check the contents

## DATA DICTIONARY:
#Dynamic inputs / Process variables - Columns 1 to 9: var.1, var.2, var.3, var.4, var.5, var.6, var.7, var.8, var.9
#These process variables are the dynamic inputs of the process that can be observed and modified by the operators of the process

#Environmental conditions / Noise factors - Columns 10 to 12: env.mbar, env.temp, env.hum
#These variables are few measures of environmental conditions that are measured at the production facility and that might impact the quality of output

#Static inputs / Material Characteristics - Columns 13 to 15: mat.1, mat.2, mat.3
#These variables are data on measurements of material inputs to the friction plate production that are measured from the incoming materials and are assessed to potentially have impactful variance

# SPLIT DATA - YOUR TASK
set.seed(2) #You can use set.seed command to control the random sample for different runs of your code

## Your task: Split the data into 70/30 split to training data set "train" and testing data set "test"

N <- nrow(friction_plate_data)

train_index <- runif(N) < .70 # generate vector of random numbers
train <-friction_plate_data[train_index,] # create the train sample based on train_index
test <- friction_plate_data[!train_index,] # assign rest to the test sample using the not operator ! 


head(train)

# NORMALIZATION - To train the Neural network model properly we need to normalize the input variables to an equal range

#Usually normalization is done to [0,1] or [-1,1] intervals, min-max can be used to scale the data to intervals [0,1]
max_vals <- apply(friction_plate_data[,-ncol(friction_plate_data)], MARGIN=2, max) #We normalize all numeric value variables, the Test.Result is the last column and is dropped from the normalization (note that there are numerous ways to do subsetting data frames). MARGIN=2 means that we want return values as column max-values
min_vals <- apply(friction_plate_data[,-ncol(friction_plate_data)], MARGIN=2, min)

nums_scaled <- as.data.frame(scale(friction_plate_data[,-ncol(friction_plate_data)], center= min_vals, scale = max_vals - min_vals)) #Scaled numeric values. Scale returns a matrix, which is coerced into data.frame object

train_nums_scaled <- as.data.frame(scale(train[,-ncol(train)], center= min_vals, scale = max_vals - min_vals)) #Scaled numeric values. Scale returns a matrix, which is coerced into data.frame object
test_nums_scaled <- as.data.frame(scale(test[,-ncol(test)], center= min_vals, scale = max_vals - min_vals)) #Scaled numeric values. Scale returns a matrix, which is coerced into data.frame object

train_scaled <- cbind(train_nums_scaled, train[, ncol(train)]); colnames(train_scaled)[ncol(train)] <- "Test.Result" #Training data scaled ,Let's combine the output column to the scaled data set 
test_scaled <- cbind(test_nums_scaled, test[, ncol(test)]); colnames(test_scaled)[ncol(train)] <- "Test.Result" #Test data scaled ,Let's combine the output column to the scaled data set 

#Other models than the Neural Network will be able to use the non-scaled input data

# MODELS FROM DATA SCIENCE TEAM - After data split done by you, you can run the code from your data team for some prediction models.
#Data science team has done some model training and tuning work and suggests using these advanced models with the given parameters for the Friction plate Test-approval prediction

# RANDOM FOREST
rf_fit <- randomForest(as.factor(Test.Result)  ~ .,data=train, importance =TRUE, maxnodes = 10) #Random Forest model training with all of the variables, package randomForest
rf_predictions <- data.frame(class = predict(rf_fit, test, type = "class"), prob = predict(rf_fit, test, type = "prob")[,2]) #In predict selection "class" gives classification TRUE/FALSE prediction and "prob" gives probability of this prediction in two columns "FALSE/TRUE"

confusionMatrix(as.factor(test$Test.Result), data=as.factor(rf_predictions$class),positive="TRUE")
par(pty="s") #Get rid of extra padding in the ROC plot
roc(as.numeric(test$Test.Result), rf_predictions$prob,  plot=TRUE, legacy.axes=TRUE, xlab="False Positive rate", ylab="True Positive rate")

importance(rf_fit)
varImpPlot(rf_fit)

reprtree:::plot.getTree(rf_fit)

# SUPPORT VECTOR MACHINE
#Support vector classifier: Find information in the course book ISLR chapter 9.3

svm_fit <- svm(as.factor(Test.Result)  ~ mat.2 + var.3 + var.7 + var.1 + var.9 + var.5 + env.mbar, #SVM was tuned and data science team selected these variables and tuning parameters for the model
               data=train, kernel="polynomial", cost=1, degree=3, coef0=2, probability = TRUE) #kernel "polynomial" for non-linear classifiers, kernel="radial" for radial kernels, and "linear" for linear kernels
svm_fit_prob <- attr(predict(svm_fit, test, probability = TRUE), "probabilities")[, 2] #To get the SVM probabilities, we need to look into attribute-property of the predict-output and call for attribute "probabilities"
svm_predictions <- data.frame(class = predict(svm_fit, test, type = "class"), prob = svm_fit_prob)  #Support vector machine predictions for the test data

confusionMatrix(as.factor(test$Test.Result), data=as.factor(svm_predictions$class),positive="TRUE")
roc(as.numeric(test$Test.Result), svm_predictions$prob,  plot=TRUE, legacy.axes=TRUE, xlab="False Positive rate", ylab="True Positive rate")


# NEURAL NETWORK
#neuralnet package information: https://cran.r-project.org/web/packages/neuralnet/neuralnet.pdf
nn_fit <- neuralnet(as.factor(Test.Result)  ~ ., hidden=c(3,2), data=train_scaled, linear.output = FALSE) #Training of the Neural Network with the parameters chosen in the data science teams's testing
#Parameters set for the Neural Network by the Data Science team
#hidden: vector of numbers specifying number of hidden neurons in each layer to be trained
#data: training data to be used
#linear.output: if the output is going to be smoothed

#NOTE for the prediction, we have to check the order of the columns in the prediction model for the response probabilities
response_col <- which(nn_fit$model.list$response == TRUE)
nn_predictions <- data.frame(class = predict(nn_fit, test_scaled[,-ncol(test_scaled)])[,response_col]<0.5, prob = predict(nn_fit, test_scaled[,-ncol(test_scaled)])[,response_col])  #Neural network prediction is a probability, we use the 0.5 decision criterion to translate the probabilities to TRUE/FALSE predictions
# !Changed the unequality sign direction to flip the obviously mismatched predictions.
#Correction to the code 19th March: The order of the classes in the predicted probabilities output from a neural network model in R depends on the order in which the levels of the response variable were specified in the original dataset. As TRUE was second in our data, we need to check the second column of probabilities.
plot(nn_fit, rep = 'best')# Plot neural network

confusionMatrix(as.factor(test$Test.Result), data=as.factor(nn_predictions$class),positive="TRUE")
roc(as.numeric(test$Test.Result), nn_predictions$prob,  plot=TRUE, legacy.axes=TRUE, xlab="False Positive rate", ylab="True Positive rate")


# BOOSTED REGRESSION TREE
#Generalized boosted models, also Gradient Boosting Machines https://cran.r-project.org/web/packages/gbm/vignettes/gbm.pdf or course book ISLR 8.2.3 or check Trevor Hastie presentation on boosting and ensemble models generally: https://www.youtube.com/watch?v=wPqtzj5VZus
#These models are very flexible and require quite a lot tuning to get a good and robust model, but can be very powerful, also prone to overfitting

#Train the best boosted model:
#The following code implements the model that the data science team found to be best through grid seach tuning
boost_fit <- gbm(as.numeric(Test.Result)  ~ .,data=train, distribution = "bernoulli", n.trees=405, interaction.depth = 3, shrinkage = 0.01, n.minobsinnode=5, bag.fraction=0.65) #Note bernoulli fit for classification, use "gaussian" for regression problems
boost_predictions <- data.frame(class = predict(boost_fit, test, n.trees = 405, type="response")>0.5, prob = predict(boost_fit, test, n.trees = 405, type="response"))  #Gradient boosting model predictions, we use the decision criteria of 0.5 again
summary(boost_fit) #See the relative importance of variables for the gradient boosting machine model

confusionMatrix(as.factor(test$Test.Result), data=as.factor(boost_predictions$class),positive="TRUE")
roc(as.numeric(test$Test.Result), boost_predictions$prob,  plot=TRUE, legacy.axes=TRUE, xlab="False Positive rate", ylab="True Positive rate")

# YOUR PREDICTION MODELS - your task


# LOGISTIC REGRESSION
logit_fit <- glm(as.factor(Test.Result) ~ ., data = train, family = binomial()) 
summary(logit_fit)
#select only the most useful variables, where p value is under 0.05
logit_fit <- glm(as.factor(Test.Result) ~  var.7 + var.9  + env.mbar + mat.2, data = train, family = binomial())  
logit_predictions <- data.frame(class = predict(logit_fit, test, type = "response"), prob = predict(logit_fit, test, type = "response"))
logit_predictions$class <- as.logical(logit_predictions$class >= 0.5)

confusionMatrix(as.factor(test$Test.Result), data=as.factor(logit_predictions$class),positive="TRUE")
roc(as.numeric(test$Test.Result), logit_predictions$prob,  plot=TRUE, legacy.axes=TRUE, xlab="False Positive rate", ylab="True Positive rate")

# DECISION TREE

dt_model <- rpart(as.factor(Test.Result) ~ ., data = train, method = "class", control = c(maxdepth = 3))
dt_predictions <- data.frame(class = predict(dt_model, test, type = "class"), prob = predict(rf_fit, test, type = "prob")[,2])

confusionMatrix(as.factor(test$Test.Result), data=as.factor(dt_predictions$class),positive="TRUE")
roc(as.numeric(test$Test.Result), dt_predictions$prob,  plot=TRUE, legacy.axes=TRUE, xlab="False Positive rate", ylab="True Positive rate")

plot(dt_model)
text(dt_model)

# MODEL PERFORMANCE - your task
#See assignment question 1 and 2

#EXAMPLE Confusion Matrix and ROC curve
#confusionMatrix(as.factor(test$Test.Result), data=as.factor(boost_predictions$class),positive="TRUE")
#roc(as.numeric(test$Test.Result), boost_predictions$prob,  plot=TRUE, legacy.axes=TRUE, xlab="False Positive rate", ylab="True Positive rate")

# SELECTION OF RECOMMENDED MODEL - your task
#See assignment question 2 and 3

#Credit to Andrzej Zuranski @ https://medium.com/@azuranski/permutation-feature-importance-in-r-randomforest-26fd8bc7a569


