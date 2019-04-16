import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.Terms;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.AttributeFactory;
import org.apache.lucene.util.Version;
import org.apache.lucene.document.Field;
import org.apache.lucene.analysis.en.PorterStemFilter;

import com.opencsv.CSVReader;


public class GenerateIndex {
	private static String pathString = "../data/";

	public static void main(String[] args) 
	{
		System.out.println("Collection creation");
//		String reviews = "/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/output/review_sub.csv";
//		String tips = "/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/IR_FinalProject/output/tip.csv";
		String reviews = pathString + "output/review_sub.csv";
//		String tips = pathString + "output/tip.csv";

		try {		
			
			Map<String, ArrayList<String>> reviewsCollection = createReviewList(reviews);
//			Map<String, ArrayList<String>> tipsCollection = createReviewList(tips);

			
			generateIndex(reviewsCollection);
//			analyseIndexes("/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/IR_FinalProject/output/indexes/review_sub/");
			analyseIndexes(pathString + "output/indexes/review_sub/");
		}
		catch(Exception e) {
			e.printStackTrace();
		}

	}
	
	public static Map<String, ArrayList<String>> createReviewList(String dataset) throws IOException {
		
		Map<String, ArrayList<String>> textCollection = new HashMap<String, ArrayList<String>>();
		
		try {
			CSVReader reader = new CSVReader(new FileReader(dataset));
			int i = 0;
		
			String[] Data;
			while((Data = reader.readNext())!= null){
				ArrayList<String> ListPerBusiness = null;
				String businessID = Data[0];
				System.out.println(businessID);
				String rawtext = Data[6];
				String text = removeStopWords(rawtext);
				System.out.println(text);
				System.out.println("Processing line: " + i++);

				if(textCollection.get(businessID) != null){
				ListPerBusiness = textCollection.get(businessID);
				ListPerBusiness.add(Data[6]);	
				}
				else{
					ListPerBusiness = new ArrayList<String>();
					ListPerBusiness.add(Data[6]);	
					textCollection.put(businessID,ListPerBusiness);
					
					}
			}
			
			System.out.println("SIZE OF REVIEW COLLECTION:" + textCollection.size());
			int count = 0;
			for (Map.Entry<String, ArrayList<String>> entry : textCollection.entrySet()) {
				String key = entry.getKey();
				ArrayList<String> value = entry.getValue();
				for (String iter : value) {
					count++;
				}
			}
			System.out.println("Final count = " + count);

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return textCollection;

	}
	
	@SuppressWarnings("resource")
	public static String removeStopWords(String str) throws IOException {
		
		ArrayList<String> stop_words = new ArrayList<String>();
//		BufferedReader br = new BufferedReader(new FileReader("/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/output/stopwords.txt"));
		BufferedReader br = new BufferedReader(new FileReader(pathString + "stopwords.txt"));
		String line = br.readLine();
		while(line != null) {
			stop_words.add(line);
			line = br.readLine();
		}
		CharArraySet stopSet = new CharArraySet(stop_words, true);
		CharArraySet stopWords = EnglishAnalyzer.getDefaultStopSet();
		
		Reader target = new StringReader(str);
		AttributeFactory factory = AttributeFactory.DEFAULT_ATTRIBUTE_FACTORY;
		StandardTokenizer tokenizer = new StandardTokenizer(factory);
		tokenizer.setReader(target);
		tokenizer.reset();
		
		StringBuilder sb = new StringBuilder();
		CharTermAttribute charTermAttribute = tokenizer.addAttribute(CharTermAttribute.class);
		while(tokenizer.incrementToken()) {
			String term = charTermAttribute.toString();
			term.replace(".", " ");
			sb.append(term + " ");
		}
 		String tokenizedReviewText = sb.toString();
		return tokenizedReviewText;
	}
	
	public static void generateIndex(Map<String, ArrayList<String>> reviewsCollection) throws IOException {
		
//		String indexPath = "/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/IR_FinalProject/output/indexes/review_sub/";
		String indexPath = pathString + "output/indexes/review_sub/";

		Directory dir = FSDirectory.open(Paths.get(indexPath));
		
		Analyzer analyzer = new StandardAnalyzer();
		IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
		iwc.setOpenMode(OpenMode.CREATE);
		IndexWriter writer = new IndexWriter(dir, iwc);
		
		List<String> businessList = new ArrayList<String>(reviewsCollection.keySet());
//		businessList.addAll(tipsCollection.keySet());
		
		Set<String> uniqueBusiness = new HashSet<String>();
		uniqueBusiness.addAll(businessList);
		businessList.clear();
		businessList.addAll(uniqueBusiness);
		
		System.out.println(businessList);
		
		for(String businessID : businessList) {
			
			if(businessID == null) {
				System.out.println("No business ID found");
				continue;
			}
		
		System.out.println("BusinessID" + businessID);
		boolean isBusinessIDExist = false;
		
		Document doc = new Document();
		
		if(reviewsCollection.containsKey(businessID)) {
			for(String review : reviewsCollection.get(businessID)) {
				
				doc.add(new TextField("businessID", businessID, Field.Store.YES));
				isBusinessIDExist = true;
				doc.add(new TextField("text", review.toLowerCase(), Field.Store.YES));
			}
		}
		
//		if(tipsCollection.containsKey(businessID)) {
//			for(String tip : tipsCollection.get(businessID)) {
//				
//				doc.add(new TextField("BusinessID", businessID, Field.Store.YES));
//				isBusinessIDExist = true;
//				doc.add(new TextField("Text", tip.toLowerCase(), Field.Store.YES));
//			}
//		}
		
		writer.addDocument(doc);
		
	}
		writer.forceMerge(1);
		writer.commit();
		writer.close();
	
	}
	
	public static void analyseIndexes(String indexPath) throws IOException {
		
		IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get((indexPath))));
		System.out.println("================================================");
		
		System.out.println("Total number of documents in the corpus: " + reader.maxDoc());
		
		Terms vocabulary = MultiFields.getTerms(reader, "text");
		
		System.out.println("Size of the vocabulary for this field: " + vocabulary.size());
		
		System.out.println("Number of documents that have at least one term for this field: " + vocabulary.getDocCount());

		System.out.println("Number of tokens for this field: " + vocabulary.getSumTotalTermFreq());
		
		System.out.println("Number of postings for this field: " + vocabulary.getSumDocFreq());
		
		System.out.println("================================================");

		System.out.println();
		reader.close();
		
		
	}
}
