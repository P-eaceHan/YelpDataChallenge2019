/**
 * code to extract and preprocess the Yelp data
 * @author Peace Han
 * @author Krupa Patel
 */
package Task1;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import com.opencsv.CSVWriter;

public class EasySearch {

	public static void main(String[] args) throws ParseException, IOException {
		EasySearch search = new EasySearch();
		
		//path to indexes generated from GenerateIndex.java
		String indexPath = "/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/IR_FinalProject/output/index/review_sub/";

		System.setProperty("file.encoding","UTF-8");
		//path to query : categories.csv
		String queryPath  = "/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/IR_FinalProject/output/categories.csv";
		
		ArrayList<String> categoriesList = new ArrayList<String>();
		
		try(BufferedReader br = new BufferedReader(new FileReader(queryPath))) {
		    for(String line; (line = br.readLine()) != null; ) {
		    	categoriesList.add(line);
				System.out.println("Category :" + line);		    }
		
		    HashMap<String, HashMap<String, Float>> businessCatMap = new HashMap<String, HashMap<String, Float>>(); 
			for(String category: categoriesList){
				System.out.println("The category is: "+category);
					
				IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths
						.get(indexPath)));
				IndexSearcher searcher = new IndexSearcher(reader);
		
				Analyzer analyzer = new StandardAnalyzer();
				QueryParser parser = new QueryParser("text", analyzer);
				
				Query query = parser.parse(category);
				Set<Term> queryTerms = new LinkedHashSet<Term>();
				searcher.createNormalizedWeight(query, false).extractTerms(queryTerms);
				
				try{
				// Get the pre-processed query terms
				EasySearch e = new EasySearch();
				e.computeRelevanceScores(queryTerms,reader,searcher, businessCatMap,category);
				}catch(IOException e){
					System.out.println("IO exception has occured");
				}catch(ParseException pe){
					System.out.println("A parse exception has occured");
				}
				
				reader.close();
			}
			
			
	}
	}		
	
		public void computeRelevanceScores(Set<Term> queryTerms,IndexReader reader, IndexSearcher searcher, HashMap<String, HashMap<String, Float>> businessCatMap,String category) throws IOException, ParseException
		{
		
		//Get query terms from the query string 
		Float relevanceScore = (float) 0.0;
		int TotalDocsInCorpus = reader.maxDoc();	
		for (Term t : queryTerms) {
			System.out.println(t.text());
		
		System.out.println();
		
		/**
		 * Get document frequency
		 */
		int df=reader.docFreq(t);
		System.out.println("Number of documents containing the term"+ t + "for field TEXT: "+df);
		System.out.println();
		
		
		double idf = Math.log10(1 + (TotalDocsInCorpus/df));
		System.out.println("IDF for term" + t + idf);
		/**
		 * Get document length and term frequency
		 */
		// Use DefaultSimilarity.decodeNormValue(…) to decode normalized document length
		ClassicSimilarity dSimi = new ClassicSimilarity();
		// Get the segments of the index
		List<LeafReaderContext> leafContexts = reader.getContext().reader()
				.leaves();
		System.out.println("LeafContext size: " + leafContexts.size());
		// Processing each segment
		for (int i = 0; i < leafContexts.size(); i++) {
			LeafReaderContext leafContext = leafContexts.get(i);
			int startDocNo = leafContext.docBase;
			
			//get frequency of term from its postings
			PostingsEnum de = MultiFields.getTermDocsEnum(leafContext.reader(), "text", new BytesRef(t.text()));
			int doc;
			if (de != null) {
				while((doc = de.nextDoc()) != PostingsEnum.NO_MORE_DOCS) {
					float normDocLeng = dSimi.decodeNormValue(leafContext.reader().getNormValues("text").get(doc));
					double termFrequency = (de.freq()/normDocLeng*normDocLeng);
					relevanceScore = (float) (termFrequency * idf);
					storeScores(searcher.doc(de.docID()).get("businessID"), relevanceScore);	
				}
				System.out.println("Relevance Score: " + relevanceScore);
				}	
				}
			}

		for(Map.Entry<String, Float> map : relevanceScores.entrySet()) {
			if(businessCatMap.containsKey(map.getKey())){
				HashMap<String, Float> hm = businessCatMap.get(map.getKey());
				hm.put(category, map.getValue());
				businessCatMap.put(map.getKey(), hm);

			}
			else{
				HashMap<String, Float> hm = new HashMap<String,Float>();
				hm.put(category, map.getValue());
				businessCatMap.put(map.getKey(), hm);
			}
			
			System.out.println("Map Key :" + map.getKey() + "and Map Value:" + map.getValue());
			
		}
		
		System.out.println("Business Category Mapping:" + businessCatMap.size());
		for(Map.Entry<String, HashMap<String,Float>> outer : businessCatMap.entrySet()){
			System.out.println("Business Id: " + outer.getKey());
			//System.out.println();
			HashMap<String, Float> innerMap = outer.getValue();
			Map<String, Float> sortedMap = sortByValue(innerMap);
//			System.out.println(sortedMap);
			int n=0;
			CSVWriter csvWriter= new CSVWriter(new FileWriter("/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/IR_FinalProject/output/index/TFIDF.csv", true));
			ArrayList<Object> arrayList=new ArrayList<>();
			System.out.println("OuterKey:" + outer.getKey());
			arrayList.add(outer.getKey());
			for(Map.Entry<String, Float> inner : sortedMap.entrySet()){
				if(n<=2){
				System.out.print("Category: " + inner.getKey() + "Value  "  + inner.getValue());
				arrayList.add(inner.getKey());
				arrayList.add(inner.getValue().toString());
				n++;
				}
			}
			Object[] a = arrayList.toArray();
			String[] res = Arrays.copyOf( a, a.length, String[].class);
			if(a[0].toString().length() < 30){
				csvWriter.writeNext(res);
				csvWriter.close();
			//System.out.println();
		}
		}
		
	}
	
	HashMap<String, Float> relevanceScores = new HashMap<String, Float>();
	
	public void storeScores(String docID, Float relevanceScore) {
		if(relevanceScores.containsKey(docID)) {
			float val = relevanceScores.get(docID);
			val = val + relevanceScore;
			relevanceScores.put(docID, val);
		}
		else {
			relevanceScores.put(docID, relevanceScore);
			}
		
	}
	
	public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue( Map<K, V> map )
	{
	    List<Map.Entry<K, V>> list =
	        new LinkedList<>( map.entrySet() );
	    Collections.sort( list, new Comparator<Map.Entry<K, V>>()
	    {
	        @Override
	        public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 )
	        {
	            return -1*(( o1.getValue() ).compareTo( o2.getValue() ));
	        }
	    } );
	
	    Map<K, V> result = new LinkedHashMap<>();
	    for (Map.Entry<K, V> entry : list)
	    {
	        result.put( entry.getKey(), entry.getValue() );
	    }
	    return result;
	}
	
	}






