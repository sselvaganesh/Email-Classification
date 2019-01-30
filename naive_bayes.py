
# -*- coding: utf-8 -*-
"""
Created on Sun 1:16 AM 02/11/2018

@author: SELVA GANESH
"""
#==========================================================#

#Import libraries

from __future__ import division
import sys as sy
from math import log
import os
from decimal import *

from nltk.stem import *




#==========================================================#

#Declare global variables

train_data_path = ' '
test_data_path = ' '

train_spam_data_path = ' '
test_ham_data_path = ' '

train_spam_file_list = [ ]
train_ham_file_list = [ ]

test_spam_file_list = [ ]
test_ham_file_list = [ ]

train_tot_file_cnt = 0

train_spam_file_cnt = 0
train_ham_file_cnt = 0

test_spam_file_cnt = 0
test_ham_file_cnt = 0

train_prior_spam = 0
train_prior_ham = 0

train_spam_unq_words = set( )
train_ham_unq_words = set( )
train_common_word = set( )

train_word_tot_cnt = 0

train_spam_unq_word_cnt = 0
train_ham_unq_word_cnt = 0

train_spam_each_word_cnt = { }
train_ham_each_word_cnt = { }

train_spam_smoothing = { }
train_ham_smoothing = { }

train_spam_words = [ ]
train_ham_words = [ ]

test_spam_identified = 0
test_ham_identified = 0
test_cannt_predict = 0

train_word_unq_cnt = 0
train_spam_tot_words = 0
train_ham_tot_words = 0

train_unq_word = ( )
test_unq_word_cnt = { }

classified_success = 0

#==========================================================#

#Function assignment
stemmer = PorterStemmer()

getcontext().prec = 28

#==========================================================#

#Special Character
spl_char = [ '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '+', ':', '"', '<', '>', '?', '/', ';', "'", '|', '\'', '[', ']', ',', '.']

stop_words = [ 'a' , 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having','he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours' ]


#==========================================================#
#List of Lambda functions


#==========================================================#

#Sparse each word in the file
def split_words(file_name, file_type):

	global train_spam_words, train_ham_words
	
	#Read the file
	file_ptr = open(file_name, 'r')
	file_data = file_ptr.read()
	file_ptr.close()

	#Remove Special Character
	for char in spl_char:	
		file_data = file_data.replace(char, ' ')

	#if Prune == yes
	if(prune=='yes'):
		for word in stop_words:
			file_data = file_data.replace(word, ' ')

	#Retrieve list of words from the file data		
	plu_words = file_data.split()

	#Do stemming
	if (file_type == 'spam'):	
		train_spam_words = train_spam_words + [ stemmer.stem(word) for word in plu_words ]
	else:
		train_ham_words  = train_ham_words + [ stemmer.stem(word) for word in plu_words ]

	
#==========================================================#

#Laplace Smoothing
def laplace_smooth():

	#global count
	for word in train_unq_word:
		train_spam_each_word_cnt[word] = train_spam_words.count(word)
		train_ham_each_word_cnt[word]  = train_ham_words.count(word)
		train_spam_smoothing[word] = (train_spam_each_word_cnt[word] + 1) / (train_spam_tot_words + train_word_unq_cnt)
		train_ham_smoothing[word]  = (train_ham_each_word_cnt[word] + 1) / (train_ham_tot_words + train_word_unq_cnt)

"""

	#Count the word occurences and calculate Smoothing then keep it in the respective dictionary
	if (file_type == 'spam'):
		for word in train_spam_unq_words:			
			train_spam_each_word_cnt[word] = train_spam_words.count(word)
			train_spam_smoothing[word] = (train_spam_each_word_cnt[word] + 1) / (train_spam_tot_words + train_word_unq_cnt)
				
	else:
		for word in train_ham_unq_words:
			train_ham_each_word_cnt[word]  = train_ham_words.count(word)
			train_ham_smoothing[word] = (train_ham_each_word_cnt[word] + 1) / (train_ham_tot_words + train_word_unq_cnt)
	
"""
#==========================================================#

#Classify the file Spam Or Ham
def classify(file_name, name):

	global test_spam_identified, test_ham_identified, test_cannt_predict

	wt_spam = float(train_prior_spam)
	wt_ham = float(train_prior_ham)
	
	#Read the file
	file_ptr = open(file_name, 'r')
	file_data = file_ptr.read()
	file_ptr.close()

	#Remove Special Character
	for char in spl_char:	
		file_data = file_data.replace(char, " ")

	#if Prune == yes
	if(prune=='yes'):
		for word in stop_words:
			file_data = file_data.replace(word, ' ')

	#Retrieve list of words from the file data		
	plu_words = file_data.split()
	test_words = [ stemmer.stem(word) for word in plu_words ]


	test_unq_word = set(test_words)
	for word in test_unq_word:	
		test_unq_word_cnt[word] = test_words.count(word)

		if word in train_unq_word:
			wt_spam = Decimal (Decimal (wt_spam) * Decimal (Decimal (train_spam_smoothing[word]) ** Decimal (test_unq_word_cnt[word])) )
			wt_ham  = Decimal (Decimal (wt_ham)  * Decimal (Decimal (train_ham_smoothing[word] ) ** Decimal (test_unq_word_cnt[word])) )
		else:
			wt_spam = Decimal (Decimal (wt_spam) * Decimal (Decimal(1) / (Decimal(train_spam_tot_words) + Decimal(train_word_unq_cnt)) ) )
			wt_ham  = Decimal (Decimal (wt_ham)  * Decimal (Decimal(1) / (Decimal(train_ham_tot_words)  + Decimal(train_word_unq_cnt)) ) )


	if(wt_spam > wt_ham):
		test_spam_identified = test_spam_identified + 1
		#print name + '\t wt_spam: '+ str(wt_spam) + '\t wt_ham: '+ str(wt_ham) + '\t ***Spam***'
		
	if(wt_spam < wt_ham):
		test_ham_identified = test_ham_identified + 1
		#print name + '\t wt_spam: '+ str(wt_spam) + '\t wt_ham: '+ str(wt_ham) + '\t ***ham***'

	if(wt_spam == wt_ham):
		test_cannt_predict = test_cannt_predict + 1
		#print name + '\t wt_spam: '+ str(wt_spam) + '\t wt_ham: '+ str(wt_ham) + '\t ***Can not predict***'


#==========================================================#

#Calculate Accuracy of Spam files
def calc_test_spam_accuracy():

	#print test_spam_identified
	accuracy = ( test_spam_identified / test_spam_file_cnt ) * 100
	print 'Test Data - Spam file Accuracy: ' + str(accuracy) 


#==========================================================#

#Calculate Accuracy of Ham files
def calc_test_ham_accuracy():

	accuracy = ( test_ham_identified / test_ham_file_cnt ) * 100
	print 'Test Data - Ham file Accuracy: ' + str(accuracy) 


#==========================================================#
#        MAIN PROGRAM
#==========================================================#

#Take folder name from input parameter
input_parm = sy.argv

#Get the training, test data path names from input parameter
train_data_path = input_parm[1]
test_data_path = input_parm[2]
prune = input_parm[3]

#Construct training, test data path names
train_spam_data_path = train_data_path + '/spam/'
train_ham_data_path = train_data_path + '/ham/'

test_spam_data_path = test_data_path + '/spam/'
test_ham_data_path = test_data_path + '/ham/'

#Get the list of file names from training data set
train_spam_file_list = os.listdir(train_spam_data_path)
train_ham_file_list  = os.listdir(train_ham_data_path)

test_spam_file_list = os.listdir(test_spam_data_path)
test_ham_file_list  = os.listdir(test_ham_data_path)

test_spam_file_cnt = len(test_spam_file_list)
test_ham_file_cnt = len(test_ham_file_list)

#Take the file counts
train_spam_file_cnt = len(train_spam_file_list)
train_ham_file_cnt  = len(train_ham_file_list) 

train_tot_file_cnt = train_spam_file_cnt + train_ham_file_cnt

#Calculate Prior values
train_prior_spam = float(train_spam_file_cnt / train_tot_file_cnt)
train_prior_ham  = float(train_ham_file_cnt / train_tot_file_cnt)

#Process each file in Training Data Set - Spam Folder
for name in train_spam_file_list: 
	file_name = train_spam_data_path + name
	split_words(file_name, 'spam')	


#Process each file in Training Data Set - Ham Folder
for name in train_ham_file_list: 
	file_name = train_ham_data_path + name
	split_words(file_name, 'ham')	


#Count the total number of words
train_spam_tot_words = len(train_spam_words)
train_spam_unq_words = set(train_spam_words)
train_spam_unq_word_cnt = len(train_spam_unq_words)

train_ham_tot_words = len(train_ham_words)
train_ham_unq_words = set(train_ham_words)
train_ham_unq_word_cnt = len(train_ham_unq_words)

#Unique word count in each file	
train_word_tot_cnt = len(train_spam_words) + len(train_ham_words)

#Common words between Spam & Ham
train_common_word = train_spam_unq_words.intersection(train_ham_unq_words)


train_unq_word = set(train_spam_words + train_ham_words)
train_word_unq_cnt = len(train_unq_word)

#Retrieve Laplace Smoothing for each word
laplace_smooth()
#laplace_smooth('spam')
#laplace_smooth('ham')

#Reset the file counter
test_spam_identified = 0
test_ham_identified = 0
test_cannt_predict = 0

#Process Spam file in Test Data
for file_name in test_spam_file_list:
	path = test_spam_data_path + file_name
	classify(path, file_name)

calc_test_spam_accuracy();
#print "Can not predict file in SPAM#: "+ str(test_cannt_predict)

#Creat an output file to write the accuracy
file_out = open('naive.txt', 'w')
file_out.write(str((test_spam_identified/test_spam_file_cnt)*100))
file_out.write('\n')

classified_success = test_spam_identified

#Reset the file counter
test_spam_identified = 0
test_ham_identified = 0
test_cannt_predict = 0

#Process Ham file in Test Data
for file_name in test_ham_file_list:
	path = test_ham_data_path + file_name
	classify(path, file_name)

calc_test_ham_accuracy();

file_out.write(str((test_ham_identified/test_ham_file_cnt)*100))
file_out.write('\n')

classified_success = classified_success + test_ham_identified
tot_accuracy = (classified_success/(test_spam_file_cnt+test_ham_file_cnt)) * 100

print "Total Accuracy: " + str(tot_accuracy)
file_out.write(str(tot_accuracy))
file_out.write('\n')

file_out.write('--------------------------\n')
file_out.close()


#print "Can not predict file in HAM#: "+ str(test_cannt_predict)

print(" *** --- End --- *** ")


#==========================================================#






