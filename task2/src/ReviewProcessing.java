import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
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
 * [ pos_score, neg_score, [JJs+contexts], [NNs+contexts] ... ]
 * Alt features..: [ pos_score, neg_score, #!s(?), LTGs ]
 *
 * Our baseline: SVM
 * Our algo: NB - global vs. local ? (also maybe CNN)
 * TODO: word embeddings (potentially with gensim)
 * RNN based model
 * CNN model (adjectives, nouns and contexts), TFIDF feature selection
 * @author Peace Han
 * @author Krupa Patel
 */

class FeatureVector {
    /**
     * output feature vector:
     * [ pos_score, neg_score, [JJs+contexts], [NNs+contexts] ... ]
     */
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

    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append(revID);
        sb.append("\t");
        sb.append(stars);
        sb.append("\t");
        sb.append(posScore);
        sb.append("\t");
        sb.append(negScore);
        sb.append("\t");
        sb.append(jjs);
        sb.append("\t");
        sb.append(nns);
        sb.append("\n");
        return sb.toString();
    }

    public String getStar() {
        StringBuilder sb = new StringBuilder();
        sb.append(revID);
        sb.append("\t");
        sb.append(stars);
        sb.append("\n");
        return sb.toString();
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
        // for [w-3, w-2, w-1, keyword, w+1, w+2, w+3]
        sb.append(context.get(0));
        sb.append(" ");
        sb.append(context.get(1));
        sb.append(" ");
        sb.append(context.get(2));
        sb.append(" ");
        sb.append(keyword);
        sb.append(" ");
        sb.append(context.get(3));
        sb.append(" ");
        sb.append(context.get(4));
        sb.append(" ");
        sb.append(context.get(5));
        // for [keyword, w-3, w-2, w-1, w+1, w+2, w+3]
//        sb.append("[");
//        sb.append(this.keyword);
//        sb.append(", ");
//        String cont = new StringBuilder(this.context.toString())
//                .deleteCharAt(0).toString(); // delete the initial "[" from context
//        sb.append(cont);
        return sb.toString();
    }
}

public class ReviewProcessing {
    static HashMap<String, Integer> sentPos = new HashMap<>(6800);
    static HashMap<String, Integer> sentNeg = new HashMap<>(6800);

    private static void processNLP(FeatureVector review) {
        // setting up StanfordCoreNLP
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
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
        String pathString = "../data/";
        String filename = "output/review_sub.csv";
        String posWords = "opinion-lexicon-English/positive-words.txt";
        String negWords = "opinion-lexicon-English/negative-words.txt";

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

        String outfilename = "task2/review_features.tsv";
        File outfile = new File(pathString + outfilename);
        FileWriter features = new FileWriter(outfile);
        String outlabelfile = "task2/review_labels.tsv";
        File labelfile = new File(pathString + outlabelfile);
        FileWriter labels = new FileWriter(labelfile);
        file = new File(pathString + filename);
        buffr = new BufferedReader(new FileReader(file));
        System.out.println("extracting feature vectors and labels from " + filename);
        line = buffr.readLine();
        while ((line = buffr.readLine()) != null) {
            String[] lineArr = line.split(",");
            System.out.println(lineArr[6]); // this is the text
            String revID = lineArr[1]; // this is the review id
            String text = lineArr[6];
            float stars = Float.parseFloat(lineArr[2]); // this is the stars rating
            FeatureVector featVec = new FeatureVector(revID, text, stars);
            processNLP(featVec);
            System.out.println(featVec.toString());
            System.out.println(featVec.getStar());
            features.write(featVec.toString());
            labels.write(featVec.getStar());
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
        features.close();
        labels.close();
        buffr.close();
    }
}
