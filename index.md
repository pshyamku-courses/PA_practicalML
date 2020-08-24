---
title: "Predicting quality of exercise movements"
author: "pshyamku-courses"
date: "8/22/2020"
output: 
  html_document: 
    keep_md: yes
---



## Executive Summary
Among health enthusiasts, it may be important to determine whether they are doing weight lifting exercises properly on every repetitive motion. The data set provide by HAR http://groupware.les.inf.puc-rio.br/har is a very useful resource to build a tool that could potentially guide exercise movements to maximize returns in terms of muscle strength. The goal of this effort is to develop a model for prediction of quality of movements for an exercise like bicep curls.

Several models were evaluated and the best performing models were found to be the random forest(rf) and extreme gradient boosting (XGboost) models.

## Load required libraries


```r
library(caret, warn.conflicts = FALSE, quietly = TRUE)
library(knitr, warn.conflicts = FALSE, quietly = TRUE)
library(doParallel, warn.conflicts = FALSE, quietly = TRUE)
# We want to parallelize over all available cores to minimize computation time
cl <- makePSOCKcluster(detectCores(all.tests = FALSE, logical = TRUE))
registerDoParallel(cl)
```

# Download the data if needed and load it


```r
# Check if data is present in the working directory
if (!file.exists("./Data")) {
  dir.create("./Data")
}
training_dataset_url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_dataset_url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
# Download the data if it is not present
if (!file.exists("./Data/pml-training.csv")) {
  download.file(training_dataset_url, destfile="./Data/pml-training.csv")
}
if (!file.exists("./Data/pml-testing.csv")) {
  download.file(test_dataset_url, destfile="./Data/pml-testing.csv")
}
# Load the data
training_all <- read.csv(file = "./Data/pml-training.csv")
testing_all <- read.csv(file = "./Data/pml-testing.csv")
# We convert the classe variable in to a factor variable suitable for classification models.
training_all$classe <- as.factor(training_all$classe)
```

## Exploration and cleaning
Following data cleaning operations are being performed on the test and training data sets -

* Both the training and test data sets have several columns that are filled with NAs.These columns have values only when a new measurement is created because they are summary statistics. Under the assumption that no information is lost for model training solely by removing the summary statistics for each time window, we remove these columns from the data set. 
* The time stamp related columns **raw_time_part_1**, **raw_timestamp_part_2**, **cvtd_timestamp**, **new_window**, and **num_window** are being dropped because the time of occurrence is not likely to have shared information with the performance of the movement.
* The variable **user_name** corresponding to the subject's name should not influence the model output so it is being removed as well.


```r
cleanDataSet <- function(data_set) {
  # Find and remove summary statistics columns
  columns_summary_statistics <- grep("^(kurtosis_|skewness_|max_|min_|amplitude|var_|avg_|stddev_|total_)", names(data_set))
  cleaned_data_set <- data_set[,-columns_summary_statistics]
  # Find and remove time stamps columns
  columns_time_data <- grep("^(raw_timestamp_|cvtd_timestamp|new_window|num_window)", names(data_set))
  cleaned_data_set <- cleaned_data_set[,-columns_time_data]
  # Find and remove subject names and serial numbers
  columns_username_serial_number <- grep("^(user_name|X)", names(data_set))
  cleaned_data_set <- cleaned_data_set[, -columns_username_serial_number]
  cleaned_data_set
}

training <- cleanDataSet(training_all)
testing <- cleanDataSet(testing_all)
```

We will set k=5 for our K-folds cross validation, resulting in each fold being ~ 20% of the number of observations in the training set. This level of crossfolds is both computationally reasonable and able to provide accurate models with lower variance.

```r
set.seed(8232020)
# List of models that we will evaluate 
models_to_evaluate <- c("rf","gbm","xgbTree","lda","rpart")
num_cross_folds <- 5 # Number of cross validation folds
# initialize arrays to hold performance results
modelFit <- list(length = length(models_to_evaluate))
modelFit_accuracy <- vector(length = length(models_to_evaluate))
# Iterate over the different types of models
for(i in 1:length(models_to_evaluate)){
  if (models_to_evaluate[i] == "rf"){
    mdl <- train(classe ~ ., data=training, method=models_to_evaluate[i], 
                 na.action = na.omit, verbose = FALSE, ntree=200, 
                 trControl=trainControl(method='cv', 
                                        number=num_cross_folds,  savePredictions = "final"))
  } else {
    mdl <- train(classe ~ ., data=training, method=models_to_evaluate[i], 
                 na.action = na.omit, 
                 trControl=trainControl(method='cv', 
                                        number=num_cross_folds,  savePredictions = "final"))
  }
  modelFit[[i]] <- mdl
  modelFit_accuracy[i] <- mdl$results$Accuracy[which.max(mdl$results$Accuracy)]
}
```

We see from the table below that based on the in sample accuracy, the rf model and the XGboost models perform almost the same and are better than the other models. GBM model also performs nearly as well as rf and XGboost.

```r
# Tabulate the results
kable(data.frame(Model = models_to_evaluate, inSample.Accuracy = modelFit_accuracy))
```



|Model   | inSample.Accuracy|
|:-------|-----------------:|
|rf      |         0.9943429|
|gbm     |         0.9622873|
|xgbTree |         0.9958719|
|lda     |         0.6935071|
|rpart   |         0.5145741|

We now try to compare the estimates for out of sample error in the models. The out of sample error is estimated to be the same as the out of fold error in the training data set. To determine this metric, we determine the fold of the cross validation that provided the highest accuracy for each model. We then extract the corresponding cross validation (cv) training and validation set from the training data set. We can then compute the out of sample error by comparing the predicted classes from the cv validation set to the actual classes in the cv validation set (out of fold data)

```r
EstimateOutOfSampleErrorRate <- function(mdl){
  indices <- grep(paste0('Fold',which.max(mdl$resample$Accuracy)),mdl$pred$Resample)
  testing_fold <- training[-indices,]
  (1 - confusionMatrix(predict(mdl, testing_fold[,-ncol(testing_fold)]), testing_fold$classe)$overall['Accuracy'])
}
kable(data.frame(Model = c("Random Forest", "Extreme gradient Boosting", "Gradient Boosting"),
                 Out.Of.Sample.Error.Estimate = 
                   c(EstimateOutOfSampleErrorRate(modelFit[[grep("rf",models_to_evaluate)]]),
                    EstimateOutOfSampleErrorRate(modelFit[[grep("xgbTree",models_to_evaluate)]]),
                    EstimateOutOfSampleErrorRate(modelFit[[grep("gbm",models_to_evaluate)]]))))
```



|Model                     | Out.Of.Sample.Error.Estimate|
|:-------------------------|----------------------------:|
|Random Forest             |                    0.0000000|
|Extreme gradient Boosting |                    0.0000000|
|Gradient Boosting         |                    0.0280928|

## Conclusion
We choose the rf model as the final model. However, the XGboost model is very close in terms of performance. Overall, the very high accuracy rate for both models may indicate over-fitting of the data. Nonetheless, they vastly out perform LDA algorithm and Regression Partition algorithms. More complex models are performing better in general due to the inherent complex non-linear relationship between the predictors and the target classes. Finally, we make predictions on the test data set.

```r
mdl <- modelFit[[grep("rf",models_to_evaluate)]]
predict(mdl, testing)
```
