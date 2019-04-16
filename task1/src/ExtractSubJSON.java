/**
 * code to extract and preprocess the Yelp data
 * @author Peace Han
 * @author Krupa Patel
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.json.simple.JSONObject;
import org.json.simple.JSONArray;
import org.json.simple.parser.*;

public class ExtractSubJSON {

    public static void main(String[] args) throws Exception {
        JSONParser parser = new JSONParser();
        /* Krupa file paths
        String pathString = "/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/yelp_dataset/";
        String fileString = "business.json";
        String outString = "/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/output/business_sub.csv";
        File file = new File(pathString+fileString);
        */
        /* Peace file paths */
        String pathString = "../data/";
        String fileString = "business.json";
        String outString = pathString+ "business_sub.csv";
        File file = new File(pathString+fileString);

        System.out.println("Reading in " + fileString);
        BufferedReader buffr = new BufferedReader(new FileReader(file));
        String line;
        HashMap<String, Integer> cats = new HashMap<>();
        int docCount = 0;
        // Krupa file path
//        PrintWriter pw1 = new PrintWriter("/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/output/categories.csv");
        PrintWriter pw1 = new PrintWriter(pathString + "output/categories.csv");
        StringBuilder sb = new StringBuilder();

        System.out.println("collecting categories...");
        while ((line = buffr.readLine()) != null) {
            JSONObject json = (JSONObject) parser.parse(line);
            String catString = (String) json.get("categories");
            if (catString != null){
                String[] busCats = catString.split(",");
                for (String cat : busCats) {
                    cat = cat.trim();
                    if (!cats.containsKey(cat))
                        cats.put(cat,0);
                    cats.put(cat, cats.get(cat) + 1);
                    docCount++;
                }
            }
        }
        System.out.println("Number of categories :" + cats);

        System.out.println("Sorting businesses...");
        int k = 100;
        Map<String, Integer> topK =
                cats.entrySet().stream()
                    .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                    .limit(k)
                    .collect(Collectors.toMap(
                            Map.Entry::getKey, Map.Entry::getValue,
                            (e1,e2) -> e1,
                            LinkedHashMap::new));

        /*
        int top100Count = 0;
        for (Map.Entry<String, Integer> entry : top100.entrySet()) {
            String key = entry.getKey();
            Integer val = entry.getValue();
            top100Count += val;
            System.out.println(key + ": " + val);
        }
        */

        System.out.println("total number of businesses in " + fileString + ": " + docCount);
        buffr.close();

        File outputfile = new File(outString);
        System.out.println("Reading in " + fileString + " to generate " + outString);
        buffr = new BufferedReader(new FileReader(file));
        PrintWriter pw = new PrintWriter(outputfile);
//        ArrayList<String> businessIds = new ArrayList<>();
        HashMap<String, Integer> businessIds = new HashMap<>(382000);
        HashMap<String, Integer> categories = new HashMap<>(382000);
       
        
        
        System.out.println("collecting business subset...");
        int n = 3; // number of category labels a business should have
        while ((line = buffr.readLine()) != null) {
            JSONObject json = (JSONObject) parser.parse(line);
            String catString = (String) json.get("categories");
            long review_count = (long) json.get("review_count");
            double stars = (double) json.get("stars");
            String city = (String) json.get("state");
            if (catString != null){
                String[] busCats = catString.split(",");
                if (busCats.length > 3) 
                {
                    for (String cat : busCats) {
                        cat = cat.trim();
                        if (topK.containsKey(cat) && review_count > 75) {
                        	if( stars == 1 || stars == 5) {
                            System.out.println("Review Count: " + review_count);
                            System.out.println("Stars : " + stars);
                        	if(!sb.equals(cat)) {
                        	sb.append(cat);
                        	sb.append("\n");
                        	}
                            businessIds.put((String) json.get("business_id"), 1);
                            pw.write(json.toString());
                            pw.println();
                            break;
                        	}
                        }
                        
                    }
                }
                
            }
        }
        System.out.println("JSON "+ sb.toString());
        pw1.write(sb.toString());
//        pw1.println();
        pw.close();
        pw1.close();
        buffr.close();

        
        System.out.println("total number of businesses in subset: " + businessIds.size());

        

        // now use newly generated subset of business.json to sample from review.json
        parser = new JSONParser();
        fileString = "review.json";
//        outString = "/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/output/review_sub.json";
        outString = pathString + "review_sub.json";
        file = new File(pathString+fileString);
        outputfile = new File(outString);
        pw = new PrintWriter(outputfile);

        System.out.println("Reading in " + fileString + " to generate " + outString);
        buffr = new BufferedReader(new FileReader(file));
        int revCount = 0;
        int revSubCount = 0;
        while ((line = buffr.readLine()) != null) {
//            System.out.println(line);
            JSONObject json = (JSONObject) parser.parse(line);
            String busId = (String) json.get("business_id");
            revCount++;
            if (busId != null && businessIds.containsKey(busId)){
//                System.out.println(line);
                revSubCount++;
                pw.write(json.toString());
                pw.println();
            }
        }
        
        System.out.println("total number of reviews in subset: " + revSubCount);
        System.out.println("total number of reviews in review.json: " + revCount);
        pw.close();
        buffr.close();
    }
}

