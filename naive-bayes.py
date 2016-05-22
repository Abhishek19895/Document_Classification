__author__ = 'abhisheksingh29895'

#I have used the below urls to help write this scrpt
"http://stackoverflow.com/questions/5563193/reading-from-multiple-txt-files-strip-data-and-save-to-xls"
"http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/"

#loading the needful libraries for performing data load, sentiment analysis & sampling
import  os
import  argparse
import  re
import  random
import  math
import  glob
import  numpy
from collections import Counter

#Using collections of words from the labs
stopWords = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also',
             'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be',
	     'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear',
	     'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for',
	     'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers',
	     'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is',
	     'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may',
	     'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor',
	     'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our',
	     'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since',
	     'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then',
	     'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us',
	     've', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
	     'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet',
	     'you', 'your']

#Function for stopwords to filter them out
def  removeStopwords(data):
	corpus  =  data
#Replacing Non_aplphapbets by blank spaces
	corpus  =  re.sub("[^a-zA-Z]+",  " ",  corpus)
	corpus  =  corpus.split()
#Innitializing a Null string to store all arguments & storing words with len >2
	wordstring  =  []
	for word in corpus:
		if  ((len(word)  >  2)):
			if  (word not in stopWords):
#Picking the lower case version of the word
				wordstring.append(word.lower())
	return  wordstring


#Setting the list of file & loading them by values
positive_file  =  glob.glob("./pos/*.txt")
negative_file  =  glob.glob("./neg/*.txt")

#Shuffle the list
#Sampling the overall corpus to form a giant list of 3 elements (Test & Training)
numpy.random.shuffle(positive_file)
numpy.random.shuffle(negative_file)

#Creating a list of data chunks from Negatives
neg_split  =  []
a_n  =  negative_file[  :  int(round(len(negative_file)/3))]
b_n  =  negative_file[  int(round(len(negative_file)/3))  :  int(round(2  *  len(negative_file)/3))]
c_n  =  negative_file[  int(round(2  *  len(negative_file)/3))  :  ]
neg_split.append(a_n)
neg_split.append(b_n)
neg_split.append(c_n)

#Creating a list of data chunks from Positives
pos_split  =  []
a_p =  positive_file[  :  int(round(len(positive_file)/3))]
b_p  =  positive_file[  int(round(len(positive_file)/3))  :  int(round(2  *  len(positive_file)/3))]
c_p  =  positive_file[  int(round(2  *  len(positive_file)/3))  :  ]
pos_split.append(a_p)
pos_split.append(b_p)
pos_split.append(c_p)


#Printing the results through the for loop - 3 iterations
def  cross_validation():
	final_accuracy  =  []
	for  i  in  range(0,  3):
		iteration  =  i  +  1
		test_p  =  pos_split[i]
		test_n  =  neg_split[i]
		if  i  ==  0:
			train_p  =  pos_split[i+2]  +  pos_split[i+1]
			train_n  =  neg_split[i+2]  +  neg_split[i+1]
		elif  i  ==  1:
			train_p  =  pos_split[i-1]  +  pos_split[i+1]
			train_n  =  neg_split[i-1]  +  neg_split[i+1]
		else  :
			train_p  =  pos_split[i-1]  +  pos_split[i-2]
			train_n  =  neg_split[i-1]  +  neg_split[i-2]

#Positive data
		data_pos  =  []
		#Extracting the positive files
		for fileName  in  train_p:
			i  =  open(fileName,  "r" )
			i  =  i.read()
			data_pos.append(i)
#Negative data
		data_neg  =  []
	#Extracting the negative files
		for fileName  in  train_n:
			j  =  open(fileName,  "r" )
			j  =  j.read()
			data_neg.append(j)

#Removing stopwords
		clean_data_neg  =  []
		for  i  in  data_neg:
			word  =  removeStopwords(i)
			clean_data_neg.append(word)
		clean_data_pos =  []
		for  i  in  data_pos:
			word  =  removeStopwords(i)
			clean_data_pos.append(word)

#Forming the big bags of positives & negatives
		str_pos  =  []
		for  a  in  range(0,  len(clean_data_pos)):
			for  b  in  range(0,  len(clean_data_pos[a])):
				k  =  clean_data_pos[a][b]
				str_pos.append(k)
		str_neg  =  []
		for  a  in  range(0,  len(clean_data_neg)):
			for  b  in range(0,  len(clean_data_neg[a])):
				k  =  clean_data_neg[a][b]
				str_neg.append(k)

#The overall string
		str  =  str_neg  + str_pos

#word-frequency & unique words
		wc_p  =  Counter(str_pos)
		wc_n  =  Counter(str_neg)
		wc  =  Counter(str)
		unique_words  =  len(wc)
		len_p  =  len(str_pos)
		total_p  =  len_p  +  unique_words  + 1
		wc  =  Counter(str)
		len_n  =  len(str_neg)
		total_n  =  len_n  +  unique_words  + 1

#word_probability for each word	
		for a in wc_p:
			wc_p[a]  =  float((wc_p[a]  +  1)  *  1.0)/total_p
		for b in wc_n:
			wc_n[b]  =  float((wc_n[b]  +  1)  *  1.0)/total_n

#Loading the data for test set for testing
	#Positive data
		test_pos  =  []
	#Extracting the positive files
		for fileName  in  test_p:
			i  =  open( fileName,  "r" )
			i  =  i.read()
			test_pos.append(i)

#Negative data
		test_neg  =  []
	#Extracting the negative files
		for fileName  in  test_n:
			j  =  open( fileName,  "r" )
			j  =  j.read()
			test_neg.append(j)

#Removing stopwords for negative & positive test sets
		clean_test_neg  =  []
		for i in test_neg:
			word  =  removeStopwords(i)
			clean_test_neg.append(word)

		clean_test_pos =  []
		for i in test_pos:
			word  =  removeStopwords(i)
			clean_test_pos.append(word)

#Calculating the document probability for test sets
#Positive test sets & positive train sets
		list_p_p  =  []
		for i  in clean_test_pos:
			list_p2  =  []
			for j in i:
				if j in wc_p:
					m  =  numpy.log(wc_p[j])
				else:
					m  =  numpy.log(float(1  *  1.0/total_p))
				list_p2.append(m)
			list_p_p.append(list_p2)

#Positive test sets & negative train sets
		list_p_n  =  []
		for i  in clean_test_pos:
			list_p2  =  []
			for j in i:
				if  j  in  wc_n:
					m  =  numpy.log(wc_n[j])
				else:
					m  =  numpy.log(float(1  *  1.0/total_n))
				list_p2.append(m)
			list_p_n.append(list_p2)

#Negative test sets & Positive train sets
		list_n_p  =  []
		for  i  in  clean_test_neg:
			list_p2  =  []
			for  j  in  i:
				if  j  in  wc_p:
					m  =  numpy.log(wc_p[j])
				else:
					m  =  numpy.log(float(1  *  1.0/total_p))
				list_p2.append(m)
			list_n_p.append(list_p2)

#Negative test sets & Negative train sets
		list_n_n  =  []
		for i  in  clean_test_neg:
			list_p2  =  []
			for j in i:
				if j in wc_n:
					m  =  numpy.log(wc_n[j])
				else:
					m  =  numpy.log(float(1  *  1.0/total_n))
				list_p2.append(m)
			list_n_n.append(list_p2)

#predicting the class for each document of test datasets:
#negatives test
		negative_result  =  []
		for d in range(0,  len(clean_test_neg)):
			if (sum(list_n_n[d])  >  sum(list_n_p[d])):
				j  =  0
			else:
				j  =  1
			negative_result.append(j)
		nncd  =  len(negative_result)  -  sum(negative_result)

#positives test
		positive_result  =  []
		for d in range(0,  len(clean_test_pos)):
			if (sum(list_p_p[d])  >  sum(list_p_n[d])):
				j  =  0
			else:
				j  =  1
			positive_result.append(j)
		npcd  =  len(positive_result)  -  sum(positive_result)
#Printing all the needed parameters of test performance
		print  "iteration %d :"  %iteration
		print  "num_pos_test_docs:",  len(test_p)
		print  "num_pos_training_docs:",  len(train_p)
		print  "num_pos_correct_docs:",  npcd
		print  "num_neg_test_docs:",  len(test_n)
		print  "num_neg_training_docs:",  len(test_n)
		print  "num_neg_correct_docs:",  nncd
#accuracy
		accuracy  =  (nncd  +  npcd)  *  100.0  /  (len(positive_result)  +  len(positive_result))
		accuracy  =  round(accuracy,  1)
		print  "accuracy: ",  accuracy,  "%"
		final_accuracy.append(accuracy)

#print the final results
	ave_accuracy  =  sum(final_accuracy)  /  len(final_accuracy)
	print "ave_accuracy: ",  round(ave_accuracy,  1),  "%"


def  parseArgument():
    """
    Code for parsing arguments
    """
#    parser = argparse.ArgumentParser(description = 'Parsing a file.')
#    parser.add_argument('-d', nargs = 1, required = True)
#    args = vars(parser.parse_args())
#    return args


def  main():
#     args = parseArgument()
#     directory = args['d'][0]
#     print directory
     cross_validation()


main()
