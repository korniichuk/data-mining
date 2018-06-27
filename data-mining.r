#!/usr/bin/Rscript
# -*- coding: utf-8 -*-
# Name: data-mining.R
# Version: 0.1a1

library(data.table)

# Data were extracted from images that were taken for the evaluation of
# an authentication procedure for banknotes
# https://archive.ics.uci.edu/ml/datasets/banknote+authentication
# Load data_banknote_authentication.txt file
url = paste('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/',
            'data_banknote_authentication.txt', sep='')
data = data.frame(fread(url))
names(data) = c('variance', 'skewness', 'curtosis', 'entropy', 'class')
