import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
import csv
import json

def read_json():
    with open('/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/output/task2/CalculatedRatings_1000.json') as json_data:
        json_list = []
        for line in json_data:
            json_dict = json.loads(line)
            json_list.append(json_dict)
    return json_list


def CalculateSVM(data):
    vectorizer = TfidfVectorizer()
    classifier = LinearSVC()
    train, test = train_test_split([(i['stars'], i['rating']) for i in data],test_size=.2,random_state=10)
    print("Train:")
    print(train)
    print("Test")
    print(test)
    x_train = vectorizer.fit_transform(i[0] for i in train)
    print(x_train)
    x_test = vectorizer.transform(i[0] for i in test)
    classifier.fit(x_train, [i[1] for i in train])
    score = classifier.score(x_test, [i[1] for i in test])
    print("SVM score:")
    print(score)


def CalculateMNB(data):
    vectorizer = TfidfVectorizer()
    classifier = MultinomialNB()
    train, test = train_test_split([(i['stars'], i['rating']) for i in data],test_size=.2,random_state=10)
    x_train = vectorizer.fit_transform(i[0] for i in train)
    x_test = vectorizer.transform(i[0] for i in test)
    classifier.fit(x_train, [i[1] for i in train])
    score = classifier.score(x_test, [i[1] for i in test])
    print("MNB score:")
    print(score)


def CalculateSVR(data):
    vectorizer = TfidfVectorizer()
    classifier = SVR(kernel='linear')
    train, test = train_test_split([(i['stars'], i['rating']) for i in data],test_size=.2,random_state=10)
    x_train = vectorizer.fit_transform(a for a,b in train)
    x_test = vectorizer.transform(a for a,b in test)
    classifier.fit(x_train, [b for a,b in train])
    score = classifier.score(x_test, [b for a,b in test])
    print("SVR :")
    print(score)


def main():
    data = read_json()
    print(data)
    if data:
        # CalculateMNB(data)
        CalculateSVM(data)
        # CalculateSVR(data)
    else:
        print("Missing Source Data")


if __name__ == '__main__':
    main()