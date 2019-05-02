# Running the CNN
Note: Due to filesize limits, all data must first be processed and obtained via
other parts of the project. See README for main project.

## Pre-Requisites:
* Follow instructions in README.md to get the appropriate feature vectors
* Download pre-trained word embeddings from http://vectors.nlpl.eu/repository/ (search for English)
    * ID 3, vector size 300, window 5 'English Wikipedia Dump of February 2017'
    * vocab size: 296630; Algo: Gensim Continuous Skipgram; Lemma: True
    * (These instructions are in word_embedings.py as well)
* Run word_embeddings.py to get the word embeddings
* Make sure all appropriate feature files (see README.md) are in cnn_data/

## Running the CNN
For each run of the CNN, be sure to update the logfile name (e.g. "exp4.txt")
and uncomment the desired feature file to run.

Once appropriate settings are selected, run the program. Results and 
metrics are printed to the output text file.