#!/usr/bin/Rscript
# -*- coding: utf-8 -*-
# Name: data-mining.r
# Version: 0.1a2

library(caret)
library(data.table)
library(dplyr)
library(PerformanceAnalytics)
library(rpart.plot)

########## 1 Data Overview ##########
# Load `data_banknote_authentication.txt` file
url = paste('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/',
            'data_banknote_authentication.txt', sep='')
df = data.frame(fread(url))
names(df) = c('variance', 'skewness', 'curtosis', 'entropy', 'class')

# Check size of `df` dataframe
nrow(df)

# Show the first part of `df` dataframe
head(df, 5)

# Show the last part of `df` dataframe
tail(df, 5)

########## 2 Summary Statistics ##########
########## 2.1 Mean ##########
print(noquote(paste0('Mean. Variance of Wavelet Transformed image: ', mean(df$variance))))
print(noquote(paste0('Mean. Skewness of Wavelet Transformed image: ', mean(df$skewness))))
print(noquote(paste0('Mean. Curtosis of Wavelet Transformed image: ', mean(df$curtosis))))
print(noquote(paste0('Mean. Entropy of image: ', mean(df$entropy))))

########## 2.2 Median ##########
print(noquote(paste0('Median. Variance of Wavelet Transformed image: ',
                     median(df$variance))))
print(noquote(paste0('Median. Skewness of Wavelet Transformed image: ',
                     median(df$skewness))))
print(noquote(paste0('Median. Curtosis of Wavelet Transformed image: ',
                     median(df$curtosis))))
print(noquote(paste0('Median. Entropy of image: ', median(df$entropy))))

########## 2.3 All-in-One ##########
print(noquote('Summary:'))
summary(select(df, -class))

########## 2.4 Correlation ##########
cor(df)

chart.Correlation(select(df, -class), histogram=TRUE)

########## 3 Graphics ##########
########## 3.1 Histograms ##########
par(mfrow=c(2,2))
hist(df$variance, main='Histogram of Variance',
     xlab='Variance of Wavelet Transformed Image')
hist(df$skewness, main='Histogram of Skewness',
     xlab='Skewness of Wavelet Transformed Image')
hist(df$curtosis, main='Histogram of Curtosis',
     xlab='Curtosis of Wavelet Transformed Image')
hist(df$entropy, main='Histogram of Entropy',
     xlab='Entropy of Image')

########## 3.2 Boxplots ##########
par(mfrow=c(2,2))
boxplot(df$variance, data=df, main='Boxplot. Variance', horizontal=TRUE)
boxplot(df$skewness, data=df, main='Boxplot. Skewness', horizontal=TRUE)
boxplot(df$curtosis, data=df, main='Boxplot. Curtosis', horizontal=TRUE)
boxplot(df$entropy, data=df, main='Boxplot. Entropy', horizontal=TRUE)

########## 4 Near Zero Variance Predictors ##########
nearZeroVar(select(df, -class), saveMetrics=TRUE)

########## 5 Linear Combinations ##########
findLinearCombos(select(df, -class))

########## 6 Highly Correlated Variables ##########
df$class = as.character(ifelse(df$class=='1', 'Y', 'N'))
df2 = select(df, -class)
cor_matrix = cor(df2)
print(noquote('Highly correlated variables:'))
summary(cor_matrix[upper.tri(cor_matrix)]) # upper triangular part of a matrix

high_cor_var = findCorrelation(cor_matrix, cutoff = 0.75) # check var above 0.75
print(noquote(paste0('Highly correlated variables: ', names(df2)[high_cor_var])))

# Delete highly correlated `skewness` column from dataframe
df2 = select(df2, -skewness)

cor_matrix = cor(df2)
print(noquote('Highly correlated variables:'))
summary(cor_matrix[upper.tri(cor_matrix)]) # upper triangular part of a matrix
df = cbind.data.frame(df2, class = df$class) # add class

########## 7 Distribution ##########
print(noquote('Distribution:'))
table(df$class)

class_freq = data.frame(table(df$class))
names(class_freq) = c('class', 'freq')
percent_chart = cbind(class_freq,
                      percent=round((class_freq$freq/sum(class_freq$freq))*100, 1))
percent_chart

slices = percent_chart$percent
lbls = c('N', 'Y')
pct = round(slices/sum(slices)*100, 1)
lbls = paste(lbls, pct) # add values of pct to labels
lbls = paste(lbls, '%', sep='') # add % char to labels
pie(slices, labels=lbls, radius=1, main='Pie Chart of Distribution',
    clockwise=TRUE)

featurePlot(x=select(df, -class), y=df$class, plot='box')

########## 8 Decision Tree ##########
rtree_set = rpart(class ~ ., df)
prp(rtree_set)

########## 9 Classification ##########
# Split the data to train and test sets
train_ind = createDataPartition(df$class, p=0.7, list=FALSE) 
data_train = data.frame(df[train_ind, ])
data_test = data.frame(df[-train_ind, ])
print(noquote('Train:'))
table(data_train$class)
print(noquote('Test:'))
table(data_test$class)

# Choose validation method for the test of model
valid_par = trainControl(method='repeatedcv', number=5, repeats=10, p=0.70, preProc='range') 

########## 9.1 SVM ##########
mod_svm = train(class ~ ., data=data_train, trControl=valid_par, method='svmRadial')
mod_svm

########## 9.2 KNN ##########
mod_knn = train(class ~. , data=data_train, trControl=valid_par, method='knn')
mod_knn

# Show summary
print(noquote('Summary:'))
mod_results = resamples(list(SVM=mod_svm, KNN=mod_knn))
summary(mod_results)

########## 9.3 SVM vs KNN ##########
bwplot(mod_results, scales=list(x=list(relation='free'), y=list(relation='free')))

# Test models
test = select(data_test, -class)
test_sum = data_test$class
mod_predict_svm = predict(mod_svm, test)
print(noquote('SMV:'))
confusionMatrix(mod_predict_svm, test_sum)
mod_predict_knn = predict(mod_knn, test)
print(noquote('KNN:'))
confusionMatrix(mod_predict_knn, test_sum)
