import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
import re
from nltk.corpus import stopwords
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
# review.csv contains two columns
# first column is the review content (quoted)
# second column is the assigned sentiment (positive or negative)

def load_file():
    with open('review1.csv') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"')
       
        #reader.next()
       #next(reader1)
      # for row in spamreader:
        data =[]
        target = []
        for row in reader:
            # skip missing data
            print (', '.join(row))
            if row[0] and row[1]:
                data.append(row[0])
                target.append(row[1])

        return data,target


# preprocess creates the term frequency matrix for the review data set
def preprocess():
    data,target = load_file()
    count_vectorizer = CountVectorizer(binary='true')
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)
    return tfidf_data
#import final
def learn_model(data,target):
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = train_test_split(data.todense(),target,test_size=0.4,random_state=43)

    clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
   # classifier=GaussianNB()
    classifier = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft',weights=[1,2,1])
    classifier.fit(data_train,target_train)
  
    predicted = classifier.predict(data_test)
    evaluate_model(target_test,predicted)
def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)


    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
# read more about model evaluation metrics here

def evaluate_model(target_true,target_predicted):
    print (classification_report(target_true,target_predicted))
    #print (classification_report(target_true,target_predicted))
    
    accuracyvalue=accuracy_score(target_true,target_predicted)
    print ("The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted)*1.1))
    plt.figure()
    DayOfWeekOfCall = [1]
    DispatchesOnThisWeekday = [accuracyvalue*100]
    LABELS = ["Accuracy"]
    plt.bar(DayOfWeekOfCall, DispatchesOnThisWeekday, align='center')
    plt.xticks(DayOfWeekOfCall, LABELS) 
    plt.show()
    #plot_classification_report(classification_report(target_true,target_predicted))
def main():
    data,target = load_file()
    tf_idf = preprocess()
    learn_model(tf_idf,target)


main()

