# YelpDataChallenge2019

## Task 1
   Predict business category using review text
   
### For each

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
