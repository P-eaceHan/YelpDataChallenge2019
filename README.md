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
    For each individual category calculates TF-IDF relevance score and stores in csv file
- PredictCategories.java :
    Compares results for category prediction using different algorithms

## Task 2
   Predict review rating(stars) from the review text using sentiment analysis

   For each review, we extract features such that we get 5 experimental conditions
* Lemma: the review text is only lemmatized and otherwise not altered.
* JJOnly: the adjectives and 3 words on either side are extracted and used to represent the whole review
* NNOnly: the nouns and 3 words on either side are extracted and used to represent the whole review
* JNMixed: reviews are represented by both their adjective+contexts and noun+contexts, in the
order in which they appear
* JNSeparate: reviews are represented in the same way as JNMixed, but adjective+contexts
and noun+contexts are listed separately in the feature vector

### Running Task 2
To get the reviews from the original dataset (review.json), first run
task1/ExtractSubJSON.java and task1/ExtractJSONtoCSV.java (see above).

Once data is obtained, run ReviewProcessing.java to extract the features. 
ReviewProcessing will produce the lemmatized review, JJOnly features, NNOnly features, 
JNMixed features, and JNSeparate features. 

Small test files are provided for testing the code (see test_data/).