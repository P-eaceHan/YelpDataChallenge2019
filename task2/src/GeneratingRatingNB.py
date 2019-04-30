from __future__ import division
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import json
import csv


class Business:
    def __init__(self, business_id, stars, score):
        self.business_id = business_id
        self.stars = stars
        self.rating = score


def get_reviews(count):
    inputfile = "/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/IR_FinalProject/output/review_sub_task2.csv"
    review_collection = open(inputfile ,'r')
    review_reader = list(csv.DictReader(review_collection))
    file_name = "/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/output/task2/CalculatedRatings_" + str(
        count) + ".csv"
    rating_file = open(file_name, 'w')
    wr = csv.writer(rating_file, dialect='excel')
    for i in range(0,count):
        row = review_reader[i]
        print(i)
        line = 0
        data = []
        match_cnt = 0
        business_id = row['business_ID']
        stars = float(row['stars'])
        text = row['text']
        if text:
            line += 1
            blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
            pos = blob.sentiment.p_pos
            neg = blob.sentiment.p_neg
            avg = pos - neg
            if avg < 0.2:
                rating = 1
            elif 0.2 < avg < 0.4:
                rating = 2
            elif 0.4 < avg < 0.6:
                rating = 3
            elif 0.6 < avg < 0.8:
                rating = 4
            elif avg > 0.8:
                rating = 5
        if stars == rating:
            match = 'MATCH'
            match_cnt += 1
        elif abs(stars - rating) == 1:
            match = 'MATCH'
            match_cnt += 1
        else:
            match = "UNMATCH"


        result = [business_id, stars, rating, pos, neg, avg, match]
        data.append(business_id)
        data.append(stars)
        data.append(rating)
        data.append(pos)
        data.append(neg)
        data.append(avg)
        data.append(match)

        wr.writerow(data)

        if line%25 == 0:
            print("Completed: "+str(line))

        print("Sentiment Score: " + business_id + " - " + "Stars:" + str(stars) + " - " + "Ratings :" + str(rating) + " - " + "POS_sentiment :" + str(pos) \
              + " - " + "NEG_sentiment:" + str(neg) + " - " + "Average:" + str(avg) + " - " + match)

        match_per = (match_cnt / line) * 100
        print("Match Percentage: "+str(match_per))


def main():
    print("Hello")
    get_reviews(2000)

main()