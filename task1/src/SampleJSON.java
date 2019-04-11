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

public class SampleJSON {

    public static void main(String[] args) throws Exception {
        JSONParser parser = new JSONParser();
        String pathString = "../data/";
        String fileString = "business.json";
        String outString = "output/business_sub.json";
        File file = new File(pathString+fileString);

        System.out.println("Reading in " + fileString);
        BufferedReader buffr = new BufferedReader(new FileReader(file));
        String line;
        HashMap<String, Integer> cats = new HashMap<>();
        int docCount = 0;

        System.out.println("collecting categories...");
        while ((line = buffr.readLine()) != null) {
//            System.out.println(line);
            JSONObject json = (JSONObject) parser.parse(line);
            String catString = (String) json.get("categories");
            if (catString != null){
                String[] busCats = catString.split(",");
                for (String cat : busCats) {
//                System.out.println(cat);
                    cat = cat.trim();
                    if (!cats.containsKey(cat))
                        cats.put(cat,0);
                    cats.put(cat, cats.get(cat) + 1);
                    docCount++;
                }
            }
        }

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

        File outputfile = new File(pathString+outString);
        System.out.println("Reading in " + fileString + " to generate " + outString);
        buffr = new BufferedReader(new FileReader(file));
        PrintWriter pw = new PrintWriter(outputfile);
//        ArrayList<String> businessIds = new ArrayList<>();
        HashMap<String, Integer> businessIds = new HashMap<>(382000);

        System.out.println("collecting business subset...");
        int n = 3; // number of category labels a business should have
        while ((line = buffr.readLine()) != null) {
//            System.out.println(line);
            JSONObject json = (JSONObject) parser.parse(line);
            String catString = (String) json.get("categories");
            if (catString != null){
                String[] busCats = catString.split(",");
                if (busCats.length > 3) {
                    for (String cat : busCats) {
                        cat = cat.trim();
//                    System.out.println(cat);
                        if (topK.containsKey(cat)) {
                            businessIds.put((String) json.get("business_id"), 1);
//                        System.out.println(json.toJSONString());
                            pw.write(json.toJSONString());
                            pw.println();
                            break;
                        }
                    }
                }
            }
        }
        System.out.println("total number of businesses in subset: " + businessIds.size());

        pw.close();
        buffr.close();

        // now use newly generated subset of business.json to sample from review.json
        parser = new JSONParser();
        fileString = "review.json";
        outString = "output/review_sub.json";
        file = new File(pathString+fileString);
        outputfile = new File(pathString+outString);
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
                pw.write(json.toJSONString());
                pw.println();
            }
        }
        System.out.println("total number of reviews in subset: " + revSubCount);
        System.out.println("total number of reviews in review.json: " + revCount);
        pw.close();
        buffr.close();
    }
}

