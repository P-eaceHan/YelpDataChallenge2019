import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.*;
//import java.lang.*;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

/**
 * Program to extract features from review text in review_sub.csv
 * review_sub.csv structure:
 * [ business_ID, review_id, stars, cool, useful, funny, text ]
 * potential methods:
 *  - Valence-Arousal (VA) measure
 *  - sentiment dictionary
 *  - LTG approach
 *  - text summarization?
 *  - dependency parse?
 * output feature vector:
 * [ #positive_words, #negative_words, #!s(?), LTGs ]
 *
 * RNN based model
 * CNN model (adjectives, nouns and contexts), TFIDF feature selection
 * @author Peace Han
 * @author Krupa Patel
 */

/**
 * A class to store the final FeatureVector info to be written/analyzed
 */
class FeatureVector {
    String revID;
    String revText;
    float stars;
    int posScore, negScore;
    List<Context> jjs; // list of adjectives and their context words
    List<Context> nns; // list of nouns and their context words

    FeatureVector(String revID, String revText, float stars){
        this.revID = revID;
        this.revText = revText;
        this.stars = stars;
    }
    void setPosScore(int score) {
        this.posScore = score;
    }
    void setNegScore(int score) {
        this.negScore = score;
    }
}

/**
 * A class for storing sentence, token indeces for identifying keywords
 */
class Index {
    int i, j;
    Index(int i, int j) {
        this.i = i;
        this.j = j;
    }
}

class Context {
    String keyword;
    List<String> context;
    Context(String keyword, List<String> context) {
        this.keyword = keyword;
        this.context = context;
    }
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Context for ");
        sb.append(this.keyword);
        sb.append(": ");
        sb.append(this.context);
        sb.append("\n");
        return sb.toString();
    }
}

public class ReviewProcessing {
    static String pathString = "../data/";
    static String filename = "output/review_sub.csv";
    static String posWords = "opinion-lexicon-English/positive-words.txt";
    static String negWords = "opinion-lexicon-English/negative-words.txt";
    static HashMap<String, Integer> sentPos = new HashMap<>(6800);
    static HashMap<String, Integer> sentNeg = new HashMap<>(6800);

    private static int processNLP(FeatureVector review) {
        // setting up StanfordCoreNLP
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,depparse");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        // TODO: make CoreNLP run for longer
        String revText = review.revText;
        CoreDocument doc = new CoreDocument(revText);
        pipeline.annotate(doc);
        List<CoreLabel> tokens = doc.tokens();
        List<CoreSentence> sentences = doc.sentences();
        int pos = 0; // positive score for this review
        int neg = 0; // negative score for this review
        int i = 0; // sentence index
        LinkedList<Index> jjInxs = new LinkedList<>();
        LinkedList<Index> nnInxs = new LinkedList<>();
        for (CoreSentence sent : sentences) {
            List<CoreLabel> toks = sent.tokens();
            int j = 0; // token index
            for (CoreLabel t: toks) {
//                System.out.println("this is the value: " + t.value());
//                System.out.println("this is the word: " + t.word());
//                System.out.println("this is the lemma: " + t.lemma());
//                System.out.println("this is the tag: " + t.tag());
//                System.out.println(t.originalText());
                // count the positive and negative scores
                if (sentPos.containsKey(t.value()) || sentPos.containsKey(t.lemma()))
                    pos++;
                if (sentNeg.containsKey(t.value()) || sentNeg.containsKey(t.lemma()))
                    neg++;
                // identify indexes of JJs/NNs
                if (t.tag().contains("JJ")){
                    // do something with i,j
                    jjInxs.add(new Index(i, j));
                } else if (t.tag().contains("NN")) {
                    nnInxs.add(new Index(i, j));
                }
                j++;
            }
            i++;
        }
        review.setNegScore(neg);
        review.setPosScore(pos);
        List<Context> jjs = buildContexts(jjInxs, sentences);
//        System.out.println("context for review " + review.revID);
//        System.out.println(jjs);
        List<Context> nns = buildContexts(nnInxs, sentences);
//        System.out.println(nns);
        review.jjs = jjs;
        review.nns = nns;
        return pos;
    }

    private static List<Context> buildContexts(LinkedList<Index> indeces,
                              List<CoreSentence> sentences) {
        List<Context> out = new LinkedList<>();
        // contexts are w-3, w-2, w-1, w+1, w+2, w+3
        for (Index index : indeces) {
            CoreSentence sentence = sentences.get(index.i);
            String keyword = sentence.tokens().get(index.j).word();
            List<String> context = new LinkedList<>();
            try {
                context.add(sentence.tokens().get(index.j-3).word());

            } catch (IndexOutOfBoundsException ex) {
                context.add("%null%");
            }
            try {
                context.add(sentence.tokens().get(index.j-2).word());
            } catch (IndexOutOfBoundsException ex) {
                context.add("%null%");
            }
            try {
                context.add(sentence.tokens().get(index.j-1).word());
            } catch (IndexOutOfBoundsException ex) {
                context.add("%null%");
            }
            try {
                context.add(sentence.tokens().get(index.j+1).word());
            } catch (IndexOutOfBoundsException ex) {
                context.add("%null%");
            }
            try {
                context.add(sentence.tokens().get(index.j+2).word());
            } catch (IndexOutOfBoundsException ex) {
                context.add("%null%");
            }
            try {
                context.add(sentence.tokens().get(index.j+3).word());
            } catch (IndexOutOfBoundsException ex) {
                context.add("%null%");
            }
            assert (context.size() == 6);
            Context c = new Context(keyword, context);
            out.add(c);
        }
        return out;
    }

    public static void main(String[] arg) throws Exception {
        System.out.println("collecting positive words...");
        File file = new File(pathString + posWords);
        BufferedReader buffr = new BufferedReader(new FileReader(file));
        int sentVal = 1;
        String line;
        while ((line = buffr.readLine()) != null) {
            if (!line.contains(";")) {
//                System.out.println(line);
                line.trim();
                sentPos.put(line,sentVal);
            }
        }
        System.out.println("collecting negative words...");
        file = new File(pathString + negWords);
        buffr = new BufferedReader(new FileReader(file));
        while ((line = buffr.readLine()) != null) {
            if (!line.contains(";")) {
//                System.out.println(line);
                line.trim();
                sentNeg.put(line,sentVal);
            }
        }
        System.out.println(sentPos.toString());
        System.out.println(sentNeg.toString());

        file = new File(pathString + filename);
        buffr = new BufferedReader(new FileReader(file));
        System.out.println("extracting feature vectors from " + filename);
        // TODO: do all the NLP analyses as one function
        line = buffr.readLine();
        while ((line = buffr.readLine()) != null) {
            String[] lineArr = line.split(",");
            System.out.println(lineArr[6]); // this is the text
            String revID = lineArr[1]; // this is the review id
            String text = lineArr[6];
            float stars = Float.parseFloat(lineArr[2]); // this is the stars rating
            FeatureVector featVec = new FeatureVector(revID, text, stars);
            processNLP(featVec);
//            int posScore = scoreSentiment(sentences, sentPos);
//            int negScore = scoreSentiment(sentences, sentNeg);
//            System.out.println("positive score for " + text + ":");
//            System.out.println(posScore);
//            System.out.println("negative score for " + text + ":");
//            System.out.println(negScore);
//            for (String elem : lineArr) {
//                System.out.println(elem);
//
//            }
        }
    }
}
