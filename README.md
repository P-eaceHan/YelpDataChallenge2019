# YelpDataChallenge2019

## Task 1
   Predict business category using review text
   
   For each business we create a Lucene document containing the reviews text. Secondly, proceed to index the Lucene documents.Therafter creating a query list consisting of all the categories from the dataset where each category will be a query term. Calculating the tf-idf scores for each query term for each business.

- ExtractSubJSON.java :
    Extracts subset data from the original Yelp dataset with the selected criteria
- ExtractJSONtoCSV.java :
    Converts the sub sampled data from JSON to CSV
- GenerateIndex.java :
    Using lucene proceeds to index generation
- EasySearch.java :
    For each individual category calculates TF-IDF relevance score ans stores in csv file
- PredictCategories.java :
    Compares results for category prediction using different algorithms

## Task 2
   Predict review rating(stars) from the review text using sentiment analysis
