
#Setting the environment
import os, sys, collections
os.environ['SPARK_HOME']="/Users/abhisheksingh29895/Desktop/programming/spark-1.6.0-bin-hadoop2.6"
sys.path.append("/Users/abhisheksingh29895/Desktop/programming/spark-1.6.0-bin-hadoop2.6/bin")
sys.path.append("/Users/abhisheksingh29895/anaconda/lib/python2.7/site-packages/")
import findspark
findspark.init()
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local").setAppName("My app").set("spark.executor.memory", "2g")
sc = SparkContext(conf=conf)

#Loading the files
path = "/Users/abhisheksingh29895/Desktop/courses/CURRENT/Advance_Machine_Learning/HW2/BX-CSV-Dump/"
ratings = sc.textFile(path + "BX-Book-Ratings.csv")
books = sc.textFile(path + "BX-Books.csv")
user = sc.textFile(path + "BX-Users.csv")




#Counting the number of rows in the Ratings set
def number_of_ratings(data):
    """
    param: Ratings Dataset
    return: (Explicit/Implicit Ratings, count)
    """
    lines = data.filter(lambda p: "User-ID" not in p)
    split_lines = lines.map(lambda x: (x.split(";")[0], x.split(";")[1], x.split(";")[2]))
    dict_data = split_lines.map(lambda x: x[2]).countByValue()
    total = sum(dict_data.values())  ;  implicit = dict_data['"0"']  ;  explicit = total-implicit
    print ""
    print "Number of Explicit Ratings are %s" %(explicit)
    print "Number of Implicit Ratings are %s" %(implicit)







#function to create a RDD with count for each ratings
def ratings_frequency(data):
    """
    param: Ratings Dataset
    return: RDD of ratings/Counts
    """
    lines = data.filter(lambda p: "User-ID" not in p)
    split_lines = lines.map(lambda x: (x.split(";")[0], x.split(";")[1], x.split(";")[2]))
    split_lines1 = split_lines.map(lambda x: (x[2],1))
    rdd_data = split_lines1.reduceByKey(lambda x, y: x + y)
    rdd_data = rdd_data.sortByKey()
    print ""
    print "An RDD with [Ratings: (Count of Ratings)] has been created"





#function to create a RDD with average ratings per city
def avg_ratings_per_city(data1, data2):
    """
    param: Ratings Dataset, User Dataset
    return: city/avg.Ratings
    """
    lines1 = data1.filter(lambda p: "User-ID" not in p)
    split_lines = lines1.map(lambda x: (x.split(";")[0], x.split(";")[1], x.split(";")[2]))
    split_lines1 = split_lines.map(lambda x: (x[0],x[2]))
    split_lines3 = split_lines1.filter(lambda x: x[1] != u'"0"')
    lines2 = data2.filter(lambda p: "User-ID" not in p)
    split_lines2 = lines2.map(lambda x: (x.split(";")[0], x.split(";")[1].split(",")[0]))
    full_data = split_lines3.join(split_lines2).collect()
    table = sc.parallelize(full_data)
    table1 = table.map(lambda x: (x[1][1].encode('utf8')[1:], int(x[1][0].encode('utf8')[1:len(x[1][0])-1])))
    table2 = table1.groupByKey().mapValues(lambda x: list(x))
    table3 = table2.map(lambda x: sum(x[1])*1.0/len(x[1]))
    print ""
    print "An RDD with [City: Avg_Ratings(Explicit)] has been created"






#function to give the city with highest number of ratings
def city_highest_number_ratings(data1, data2):
    """
    param: Ratings Dataset, User Dataset
    return: city
    """
    lines1 = data1.filter(lambda p: "User-ID" not in p)
    split_lines = lines1.map(lambda x: (x.split(";")[0], x.split(";")[1], x.split(";")[2]))
    split_lines1 = split_lines.map(lambda x: (x[0],x[2]))
    lines2 = data2.filter(lambda p: "User-ID" not in p)
    split_lines2 = lines2.map(lambda x: (x.split(";")[0], x.split(";")[1].split(",")[0]))
    full_data = split_lines1.join(split_lines2).collect()
    table = sc.parallelize(full_data)
    table1 = table.map(lambda x: (x[1][1],x[1][0]))
    dict = table1.countByKey()
    dict1 = sorted(dict.items(),key = lambda x :x[1], reverse = True)
    dict1[0][0].encode("utf8")
    print ""
    print "City with the highest number of ratings is %s" %(str(dict1[0][0])[1:])







#function to create a RDD with number of ratings per author
def ratings_per_author(data1, data2):
    """
    param: Ratings Dataset, Books Dataset
    return: ratings, author
    """
    lines1 = data1.filter(lambda p: "User-ID" not in p)
    split_lines = lines1.map(lambda x: (x.split(";")[0], x.split(";")[1], x.split(";")[2]))
    split_lines1 = split_lines.map(lambda x: (x[1],x[2]))
    split_lines3 = split_lines1.filter(lambda x: x[1] != u'"0"')
    lines2 = data2.filter(lambda p: "User-ID" not in p)
    split_lines2 = lines2.map(lambda x: (x.split(";")[0], x.split(";")[2]))
    full_data = split_lines3.join(split_lines2).collect()
    table = sc.parallelize(full_data)
    table1 = table.map(lambda x: (x[1][1],x[1][0]))
    ratings_author = table1.reduceByKey(lambda x, y: x + y)
    print ""
    print "An RDD with [author: Number_ratings(Explicit)] has been created"






#function to create a RDD with number of ratings per user
def ratings_per_user(data):
    """
    param: Ratings Dataset
    return: ratings, user
    """
    lines1 = data.filter(lambda p: "User-ID" not in p)
    split_lines = lines1.map(lambda x: (x.split(";")[0], x.split(";")[1], x.split(";")[2]))
    split_lines1 = split_lines.map(lambda x: (x[0],x[2]))
    split_lines2 = split_lines1.filter(lambda x: x[1] != u'"0"')
    ratings_user = split_lines2.reduceByKey(lambda x, y: x + y)
    print "An RDD with [User: Number_ratings(Explicit)] has been created"








"""
#Part (2) : Document Classification using Naive Bayes Classifier
"""
#Function to use the standard Naive Bayes classifier of Spark to predict Document categories
def use_naive_nayes():
    """
    Running the Naive Bayes from Spark's Mlib library
    """
    from pyspark.mllib.classification import NaiveBayes
    from pyspark.mllib.feature import HashingTF, IDF
    from pyspark.mllib.linalg import SparseVector, Vectors
    from pyspark.mllib.regression import LabeledPoint
    #loading the files
    path = "/Users/abhisheksingh29895/Desktop/courses/CURRENT/Advance_Machine_Learning/HW2/aclImdb/"
    train_pos = sc.textFile(path + "train/pos/*txt").map(lambda line: line.encode('utf8')).map(lambda line: line.split())
    train_neg = sc.textFile(path + "train/neg/*txt").map(lambda line: line.encode('utf8')).map(lambda line: line.split())
    test_pos = sc.textFile(path + "test/pos/*txt").map(lambda line: line.encode('utf8')).map(lambda line: line.split())
    test_neg = sc.textFile(path + "test/neg/*txt").map(lambda line: line.encode('utf8'))
    #TF-IDF
    tr_pos = HashingTF().transform(train_pos)  ;  tr_pos_idf = IDF().fit(tr_pos)
    tr_neg = HashingTF().transform(train_neg)  ;  tr_neg_idf = IDF().fit(tr_neg)
    te_pos = HashingTF().transform(test_pos)  ;  te_pos_idf = IDF().fit(te_pos)
    te_neg = HashingTF().transform(test_neg)  ;  te_neg_idf = IDF().fit(te_neg)
    #IDF step
    tr_pos_tfidf = tr_pos_idf.transform(tr_pos)  ;  tr_neg_tfidf = tr_neg_idf.transform(tr_neg)
    te_pos_tfidf = te_pos_idf.transform(te_pos)  ;  te_neg_tfidf = te_neg_idf.transform(te_neg)
    #Creating labels
    pos_label = [1] * 12500  ;  pos_label = sc.parallelize(pos_label)
    neg_label = [1] * 12500  ;  neg_label = sc.parallelize(neg_label)
    # Combine using zip
    train_pos_file = pos_label.zip(tr_pos_tfidf).map(lambda x: LabeledPoint(x[0], x[1]))
    train_neg_file = neg_label.zip(tr_neg_tfidf).map(lambda x: LabeledPoint(x[0], x[1]))
    test_pos_file = pos_label.zip(te_pos_tfidf).map(lambda x: LabeledPoint(x[0], x[1]))
    test_neg_file = neg_label.zip(te_neg_tfidf).map(lambda x: LabeledPoint(x[0], x[1]))
    #Joining 2 RDDS to form the final training set
    train_file = train_pos_file.union(train_neg_file)
    test_file = test_pos_file.union(test_neg_file)
    # Fitting a Naive bayes model
    model = NaiveBayes.train(train_file)
    # Make prediction and test accuracy
    predictionAndLabel = test_file.map(lambda p: (model.predict(p[1]), p[0]))
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
    print ""
    print "Test accuracy is {}".format(round(accuracy,4))







#function for cleaning Text
def process_text(record):
    """ Tokenize text and remove stop words."""
    text = record['text']
    stopWords = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also','am', 'among', 'an', 'and', 'any'
    ,'are', 'as', 'at', 'be','because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear','did', 'do', 'does'
    , 'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers'
    , 'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is','it', 'its', 'just', 'least', 'let', 'like'
    , 'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor','not', 'of', 'off', 'often'
    , 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so'
    , 'some', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'too'
    , 'twas', 'us', 've', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which','while', 'who', 'whom'
    , 'why', 'will', 'with', 'would', 'yet', 'you', 'your',  'NA',  '..........', '%', '@']
    words = [''.join(c for c in s if c not in string.punctuation) for s in text]
    no_stops = [word for word in words if word not in stopWords]
    return {'label':record['label'], 'words':no_stops}







#Function to count the words
def count_word(record, index):
    return record.features[index]






#Function to classify a test record
def classify_test_record(record, log_pos_prior, log_neg_prior, log_pos_probs, log_neg_probs):
    words = np.array(record.features)
    pos_prob = log_pos_prior + np.dot(words, log_pos_probs)
    neg_prob = log_neg_prior + np.dot(words, log_neg_probs)
    if pos_prob > neg_prob:
        return 1
    else:
        return 0






#Function to calculate probability for the given words
def calc_probability(word_count, total_words, total_unique_words):
    return float(word_count + 1) / (total_words + total_unique_words + 1)






#Function to build a Naive bayes Classifier from scratch and classify documents
def build_naive_bayes():
    """
    Building the Naive Bayes from Spark
    """
    import string, numpy as np
    from collections import Counter
    from pyspark.mllib.classification import NaiveBayes
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.regression import LabeledPoint
    #loading the files
    #path = "/Users/abhisheksingh29895/Desktop/courses/CURRENT/Advance_Machine_Learning/HW2/aclImdb/"
    path = "s3n://amldataabhi/HW2/"
    train_pos = sc.textFile(path + "train/pos/*txt").map(lambda line: line.encode('utf8'))
    train_neg = sc.textFile(path + "train/neg/*txt").map(lambda line: line.encode('utf8'))
    test_pos = sc.textFile(path + "test/pos/*txt").map(lambda line: line.encode('utf8'))
    test_neg = sc.textFile(path + "test/neg/*txt").map(lambda line: line.encode('utf8'))
    #Binding the Positive & Negatives sets
    train = train_pos.map(lambda x: {'label':1, 'text':x}).union(train_neg.map(lambda x: {'label':0, 'text':x}))
    test = test_pos.map(lambda x: {'label':1, 'text':x}).union(test_neg.map(lambda x: {'label':0, 'text':x}))
    #Processing the test
    train = train.map(process_text)  ;  test = test.map(process_text)
    #Creating a dictionary
    vocabulary_rdd = train.flatMap(lambda x: x['words']).distinct()
    vocabulary = vocabulary_rdd.collect()
    #Function to count the number of words for this
    def count_words(record, vocabulary):
        word_counts = Counter(record['words'])
        word_vector = []
        for word in vocabulary:
            word_vector.append(word_counts[word])
        label = record['label']
        features = Vectors.dense(word_vector)
        return LabeledPoint(label, features)
    #
    #Word count on each of the file
    train_data = train.map(lambda record: count_words(record, vocabulary)).repartition(16)
    test_data = test.map(lambda record: count_words(record, vocabulary)).repartition(16)
    #Using MLib model
    model = NaiveBayes.train(train_data, 1.0)
    #making our own model
    total_training = train.count()
    pos_prior = train_pos.count() * 1.0/ total_training  ;  neg_prior = 1 - pos_prior  ;  num_unique_words = len(vocabulary)
    pos_total_words = train_data.filter(lambda x: x.label == 1).map(lambda x: sum(x.features)).reduce(lambda x1, x2: x1 + x2)
    neg_total_words = train_data.filter(lambda x: x.label == 0).map(lambda x: sum(x.features)).reduce(lambda x1, x2: x1 + x2)
    vocabulary_rdd_index = vocabulary_rdd.zipWithIndex().collect()
    #Creating RDDS of the words for each category
    pos_word_counts_rdd = train_data.filter(lambda x: x.label == 1).\
    flatMap(lambda x: list(enumerate(x.features))).\
    reduceByKey(lambda x1, x2: x1 + x2).sortByKey()
    neg_word_counts_rdd = train_data.filter(lambda x: x.label == 0).\
    flatMap(lambda x: list(enumerate(x.features))).\
    reduceByKey(lambda x1, x2: x1 + x2).sortByKey()
    #Storing list of all words
    pos_word_counts = []  ;  pos_probs = []  ;  neg_word_counts = []  ;  neg_probs = []  #To store the list of all positives
    for word, index in vocabulary_rdd_index:
        word_p = train_data.filter(lambda x: x.label == 1).map(lambda x: x.features[index]).reduce(lambda x1, x2: x1 + x2)
        word_n = train_data.filter(lambda x: x.label == 0).map(lambda x: x.features[index]).reduce(lambda x1, x2: x1 + x2)
        word_prob_p = float(word_p + 1) / (pos_total_words + num_unique_words + 1)
        word_prob_n = float(word_n + 1) / (neg_total_words + num_unique_words + 1)
        pos_word_counts.append(word_count)  ;  pos_probs.append(word_prob)
        neg_word_counts.append(word_count)  ;  neg_probs.append(word_prob)
    #Creatng RDDS for each of the groups
    pos_probs_rdd = pos_word_counts_rdd.map(lambda x: calc_probability(x[1], pos_total_words, num_unique_words))
    neg_probs_rdd = neg_word_counts_rdd.map(lambda x: calc_probability(x[1], neg_total_words, num_unique_words))
    #Calculating the log of probabilities
    log_pos_prior ,  log_neg_prior  =  np.log(pos_prior),  np.log(neg_prior)
    log_pos_probs,  log_neg_probs  =  np.log(np.array(pos_probs)),  np.log(np.array(neg_probs))
    #Making classification based on conditional probabilities
    classifications = test_data.map(lambda x: classify_test_record(x, log_pos_prior, log_neg_prior, log_pos_probs, log_neg_probs))
    correct = classifications.zip(test_data.map(lambda x: x.label)).filter(lambda x: x[0] == x[1]).count()
    #Accuracy is
    accuracy = correct / test_data.count()
    print ""
    print "Test accuracy is {}".format(round(accuracy,4))







#Calling the main function to run the code
if __name__ == '__main__':
    print "******* Q.1) Part 1] Number of Ratings (Explicit / Implicit)**********"
    number_of_ratings(ratings)
    print "Done"

    print "******* Q.1) Part 2] Count of each ratings**********"
    ratings_frequency(ratings)
    print "Done"

    print "******* Q.1) Part 3] average ratings per city **********"
    avg_ratings_per_city(ratings,user)
    print "Done"

    print "******* Q.1) Part 4] city with the highest rating **********"
    city_highest_number_ratings(ratings,user)
    print "Done"

    print "******* Q.1) Part 5] city with the highest rating **********"
    ratings_per_author(ratings,books)
    ratings_per_user(ratings)
    print "Done"

    print "******* Question 1 Over, now using data from AWS for Naive Bayes **********"
    exit()
    #First SSH to the PEM file to activate the instance
    #ssh -i ~/Abhishek3.pem hadoop@ec2-54-186-36-60.us-west-2.compute.amazonaws.com
    pyspark #on EMR (Hadoop) Instance
    AWS_ACCESS_KEY_ID = "AKIAJHVZ4DT6IMUHIWLQ"
    AWS_SECRET_ACCESS_KEY = "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"
    #Enabling the hadoop path for spark
    sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
    sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)
    print "Done"

    print "******* Q.2) Part 1] Document Classification using Standard Naive Bayes **********"
    use_naive_bayes()
    print "Done"

    print "******* Q.2) Part 2] Document Classification using my own Naive Bayes **********"
    build_naive_bayes()
    print "******* I have partnered with Jason Helgren from MSAN 2016 for this task!! **********"
    print "Done"

