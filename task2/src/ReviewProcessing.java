import java.io.*;
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
    String revText; // the original text of the review, lemmatized
    float stars;
    int posScore, negScore;
    List<Context> jjs; // list of adjectives and their context words
    List<Context> nns; // list of nouns and their context words
    List<Context> jns; // list of adjectives and nouns, with contexts, in order of appearance

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


    public String printJJs(){
        return jjs.toString().replace(",", "")
                .replace("[", "")
                .replace("]", "")
                .trim();
    }

    public String printNNs(){
        return nns.toString().replace(",", "")
                .replace("[", "")
                .replace("]", "")
                .trim();
    }

    public String printJNs(){
        return jns.toString().replace(",", "")
                .replace("[", "")
                .replace("]", "")
                .trim();
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
        // uncomment below for [keyword, w-3, w-2, w-1, w+1, w+2, w+3]
        /*
        sb.append("[");
        sb.append(this.keyword);
        sb.append(", ");
        String cont = new StringBuilder(this.context.toString())
                .deleteCharAt(0).toString(); // delete the initial "[" from context
        sb.append(cont);*/
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
        LinkedList<String> lemmadRevText = new LinkedList<>();
        LinkedList<Index> jjInxs = new LinkedList<>();
        LinkedList<Index> nnInxs = new LinkedList<>();
        LinkedList<Index> jnInxs = new LinkedList<>();
        for (CoreSentence sent : sentences) {
            List<CoreLabel> toks = sent.tokens();
            // get rid of punctuation
//            String punctTags = "!\"\"''#$%&()-LRB-RRB-*+,-./:;<=>?@[\\]^_``{|}~\t\n";
//            toks.removeIf(tok -> punctTags.contains(tok.tag()));
            int j = 0; // token index
//            System.out.println(sent.text());
            for (CoreLabel t: toks) {
//                System.out.println("this is the value: " + t.value());
//                System.out.println("this is the word: " + t.word());
//                System.out.println("this is the lemma: " + t.lemma());
//                System.out.println("this is the tag: " + t.tag());
//                System.out.println(t.originalText());
                // get the lemma of each token
                String lemma = t.lemma();
                lemmadRevText.add(lemma);
                // count the positive and negative scores
                if (sentPos.containsKey(t.value()) || sentPos.containsKey(t.lemma()))
                    pos++;
                if (sentNeg.containsKey(t.value()) || sentNeg.containsKey(t.lemma()))
                    neg++;
                // identify indexes of JJs/NNs
                if (t.tag().contains("JJ")){
                    jjInxs.add(new Index(i, j));
                    jnInxs.add(new Index(i, j));
                } else if (t.tag().contains("NN")) {
                    nnInxs.add(new Index(i, j));
                    jnInxs.add(new Index(i, j));
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
        List<Context> jns = buildContexts(jnInxs, sentences);
        review.jjs = jjs;
        review.nns = nns;
        review.jns = jns;
        review.revText = lemmadRevText
                            .toString()
                            .replace(",", "")
                            .replace("[", "")
                            .replace("]", "")
                            .trim();
    }

    private static List<Context> buildContexts(LinkedList<Index> indeces,
                              List<CoreSentence> sentences) {
        List<Context> out = new LinkedList<>();
        // contexts are w-3, w-2, w-1, w+1, w+2, w+3
        for (Index index : indeces) {
            CoreSentence sentence = sentences.get(index.i);
            String keyword = sentence.tokens().get(index.j).lemma();
            // get rid of punctuation

            List<String> context = new LinkedList<>();
            try {
                context.add(sentence.tokens().get(index.j-3).lemma());

            } catch (IndexOutOfBoundsException ex) {
                context.add("%null%");
            }
            try {
                context.add(sentence.tokens().get(index.j-2).lemma());
            } catch (IndexOutOfBoundsException ex) {
                context.add("%null%");
            }
            try {
                context.add(sentence.tokens().get(index.j-1).lemma());
            } catch (IndexOutOfBoundsException ex) {
                context.add("%null%");
            }
            try {
                context.add(sentence.tokens().get(index.j+1).lemma());
            } catch (IndexOutOfBoundsException ex) {
                context.add("%null%");
            }
            try {
                context.add(sentence.tokens().get(index.j+2).lemma());
            } catch (IndexOutOfBoundsException ex) {
                context.add("%null%");
            }
            try {
                context.add(sentence.tokens().get(index.j+3).lemma());
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
        long startTime = System.nanoTime();
        String pathString = "../data/";
        String filename = "task2/review_sub_task2.csv";
//        String filename = "task2/test_reviews.csv";
        /*
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
                line = line.trim();
                sentPos.put(line,sentVal);
            }
        }
        System.out.println("collecting negative words...");
        file = new File(pathString + negWords);
        buffr = new BufferedReader(new FileReader(file));
        while ((line = buffr.readLine()) != null) {
            if (!line.contains(";")) {
//                System.out.println(line);
                line = line.trim();
                sentNeg.put(line,sentVal);
            }
        }
        System.out.println(sentPos.toString());
        System.out.println(sentNeg.toString());
        */

        // raw text
        String rawString = "task2/features4.0/rawText_lemmatized.tsv";
        File rawFile = new File(pathString + rawString);
        PrintWriter rawText = new PrintWriter(rawFile);

        // JJ contexts only
        String jjString = "task2/features3.0/jjOnly.tsv";
        File jjFile = new File(pathString + jjString);
        PrintWriter jjFeatures = new PrintWriter(jjFile);

        // NN contexts only
        String nnString = "task2/features3.0/nnOnnly.tsv";
        File nnFile = new File(pathString + nnString);
        PrintWriter nnFeatures = new PrintWriter(nnFile);

        // JJ+NN, in order
        String jnMixString = "task2/features3.0/jnMix.tsv";
        File jnMixFile = new File(pathString + jnMixString);
        PrintWriter jnMixFeatures = new PrintWriter(jnMixFile);

        // JJ+NN, separate
        String jnSepString = "task2/features3.0/jnSep.tsv";
        File jnSepFile = new File(pathString + jnSepString);
        PrintWriter jnSepFeatures = new PrintWriter(jnSepFile);

        File file = new File(pathString + filename);
        BufferedReader buffr = new BufferedReader(new FileReader(file));
        System.out.println("extracting feature vectors and labels from " + filename);
        String line = buffr.readLine();
        while ((line = buffr.readLine()) != null) {
            String[] lineArr = line.split(",");
            System.out.println(lineArr[6]); // this is the text
            String revID = lineArr[1]; // this is the review id
            String text = lineArr[6];
            float stars = Float.parseFloat(lineArr[2]); // this is the stars rating

            // write the reviewID
            rawText.write(revID);
            jjFeatures.write(revID);
            nnFeatures.write(revID);
            jnMixFeatures.write(revID);
            jnSepFeatures.write(revID);

            rawText.write("\t");
            jjFeatures.write("\t");
            nnFeatures.write("\t");
            jnMixFeatures.write("\t");
            jnSepFeatures.write("\t");

            // write the correct rating
            rawText.write(Float.toString(stars));
            jjFeatures.write(Float.toString(stars));
            nnFeatures.write(Float.toString(stars));
            jnMixFeatures.write(Float.toString(stars));
            jnSepFeatures.write(Float.toString(stars));

            rawText.write("\t");
            jjFeatures.write("\t");
            nnFeatures.write("\t");
            jnMixFeatures.write("\t");
            jnSepFeatures.write("\t");

            FeatureVector featVec = new FeatureVector(revID, text, stars);
            processNLP(featVec);

            rawText.write(featVec.revText);
            jjFeatures.write(featVec.printJJs());
            nnFeatures.write(featVec.printNNs());
            jnMixFeatures.write(featVec.printJNs());
            jnSepFeatures.write(featVec.printJJs());
            jnSepFeatures.write(" ");
            jnSepFeatures.write(featVec.printNNs());

            rawText.write("\n");
            jjFeatures.write("\n");
            nnFeatures.write("\n");
            jnMixFeatures.write("\n");
            jnSepFeatures.write("\n");

//            System.out.println(featVec.toString());
//            System.out.println(featVec.getStar());
//            features.write(featVec.toString());

//            labels.write(featVec.getStar());
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
        jjFeatures.close();
        nnFeatures.close();
        jnMixFeatures.close();
        jnSepFeatures.close();
        rawText.close();
        buffr.close();
        long endTime = System.nanoTime();
        long duration = (endTime - startTime);
        System.out.println("completed in " + duration + "ns");
    }
}
