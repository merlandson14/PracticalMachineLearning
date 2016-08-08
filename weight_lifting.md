# Analysis of Barbell Workouts
M Erlandson  
August 6, 2016  

# Summary and Objective

This project will analyze variables within the Weight Lifting Exercises dataset and attempt to predict the classe outcome of "correct" or "incorrect" on performing the weight lifting of the test subjects.

## Load Data

Training Data was loaded from this file:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
Testing Data was loaded from this file:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
Information about the dataset is found on this website, under the "Weight Lifting Exercise Dataset" heading:
http://groupware.les.inf.puc-rio.br/har

All calculations done in R 3.2.2 and RStudio 0.99.473 and Windows 10.
Libraries used are ggplot2, caret, rpart, and randomForest.





Training dataset contains **19622 observations** from 6 test subjects performing repetitions of dumbell curls. Sensors in 4 places (arm, forearm, belt, and dumbbell) recorded **almost 160 variables**, like roll, pitch, yaw, acceleration, and gyros at XYZ positions. Most have number or integer values, but a few are factor variables due to NAs and !Div/0 errors. The outcome variable is **"classe"** with designations **"A"** (correct specs followed), **"B"** (elbows to front), **"C"** (lift halfway), **"D"** (lower halfway), and **"E"** (hips to front). 

## Exploratory Analysis

Let's look at the data and determine which factors we wish to use. We need to take out the non-important variables (X, username, time stamps, windows) and all those that have lots of NAs (kurtosis, max and min specific values, amplitudes). We are left with 52 possible predictor variables, plus the outcome. We also need to do the same thing on the test dataset for later.


```r
set.seed(133)
pmlTrainSub <- subset(pmlTraining, select=c(8:11,37:49,60:68,84:86,102,113:124,140,151:160))
pmlTestSub <- subset(pmlTesting, select=c(8:11,37:49,60:68,84:86,102,113:124,140,151:160))
inTrain <- createDataPartition(pmlTrainSub$classe, p=0.75, list=FALSE)
pmlTrainSet <- pmlTrainSub[inTrain,]
pmlValSet <- pmlTrainSub[-inTrain,]
```

We also want to be able to cross-validate and find out-of-bag error rates before running the test set, so we divided the training set up 75% for training and 25% for validation.

Let's look at some variables in qplots to see if we find any trends.


```r
qplot(total_accel_belt, pitch_arm, data=pmlTrainSet, colour=classe)
```

![](weight_lifting_files/figure-html/plots-1.png)

```r
qplot(accel_arm_z, pitch_forearm, data=pmlTrainSet, colour=classe)
```

![](weight_lifting_files/figure-html/plots-2.png)

```r
qplot(roll_dumbbell, pitch_dumbbell, data=pmlTrainSet, colour=classe)
```

![](weight_lifting_files/figure-html/plots-3.png)

Hmmm, lots of information all over the place, but no descernable trends. We will have to let the model fit itself against all 52 predictors.

## Model Fitting


```r
modelTree <- train(classe ~ ., method = "rpart", data = pmlTrainSub)
print(modelTree$finalModel)
```

```
## n= 19622 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 19622 14042 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 130.5 17977 12411 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -33.95 1578    10 A (0.99 0.0063 0 0 0) *
##      5) pitch_forearm>=-33.95 16399 12401 A (0.24 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 439.5 13870  9953 A (0.28 0.18 0.24 0.19 0.11)  
##         20) roll_forearm< 123.5 8643  5131 A (0.41 0.18 0.18 0.17 0.061) *
##         21) roll_forearm>=123.5 5227  3500 C (0.077 0.18 0.33 0.23 0.18) *
##       11) magnet_dumbbell_y>=439.5 2529  1243 B (0.032 0.51 0.043 0.22 0.19) *
##    3) roll_belt>=130.5 1645    14 E (0.0085 0 0 0 0.99) *
```

We tried running the model as a classification tree, since we are trying to predict classes, but it only fitted A, B, C, and E. There was no output level for D, so this is not a good model to use for prediction. [Note: I ran it on full training set before realizing I needed to split for validation, but it's not a good model, so I didn't bother to rerun it on the split set.]

Instead, let's use random forests. They are very accurate (but computationally expensive) and have the benefit of internal cross-validation. Default mtry is square root of the number of predictors, in this case about 7, but we will let the train function determine the best fit. Ntrees is defaulted to 500.


```r
mtry <- 2
modelRF <- train(classe ~ ., method="rf", data=pmlTrainSet)
```

```r
print(modelRF)
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9896904  0.9869531
##   27    0.9901078  0.9874819
##   52    0.9800812  0.9747940
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
print(modelRF$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.57%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4180    3    2    0    0 0.001194743
## B   19 2822    7    0    0 0.009129213
## C    0   11 2547    9    0 0.007791196
## D    0    0   20 2389    3 0.009535655
## E    0    1    3    6 2696 0.003695492
```

According to this print-out, the classification should work very well, with only classes "B" and "D" having at most 1% in-sample error rate. The out-of-sample error rate on the validation and test sets will be greater, especially since random forests tend to overfit to the training data.

## Validation and Test


```r
predRF <- predict(modelRF, pmlValSet)
confusionMatrix(predRF, pmlValSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    8    0    0    0
##          B    0  940    6    0    0
##          C    1    1  841   11    2
##          D    0    0    8  792    0
##          E    1    0    0    1  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.992           
##                  95% CI : (0.9891, 0.9943)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9899          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9905   0.9836   0.9851   0.9978
## Specificity            0.9977   0.9985   0.9963   0.9980   0.9995
## Pos Pred Value         0.9943   0.9937   0.9825   0.9900   0.9978
## Neg Pred Value         0.9994   0.9977   0.9965   0.9971   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2841   0.1917   0.1715   0.1615   0.1833
## Detection Prevalence   0.2857   0.1929   0.1746   0.1631   0.1837
## Balanced Accuracy      0.9981   0.9945   0.9900   0.9916   0.9986
```

So our trees fit very well with an accuracy of 99% and the OOB error rate of 1%. 

Below is the run on the test set, which we will submit for grading. Here are our answers for the twenty test scenarios.


```r
predRFtest <- predict(modelRF, pmlTestSub)
predRFtest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
