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
        File outputfile = new File(pathString+outString);

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
        Map<String, Integer> top100 =
                cats.entrySet().stream()
                    .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                    .limit(100)
                    .collect(Collectors.toMap(
                            Map.Entry::getKey, Map.Entry::getValue,
                            (e1,e2) -> e1,
                            LinkedHashMap::new));

        int top100Count = 0;
        for (Map.Entry<String, Integer> entry : top100.entrySet()) {
            String key = entry.getKey();
            Integer val = entry.getValue();
            top100Count += val;
            System.out.println(key + ": " + val);
        }

        System.out.println("total number of businesses in " + fileString + ": " + docCount);
        System.out.println("total number of businesses with labels in top100: " + top100Count);
        buffr.close();

        System.out.println("Reading in " + fileString);
        BufferedReader buffr2 = new BufferedReader(new FileReader(file));
        PrintWriter pw = new PrintWriter(outputfile);
        int busCount = 0;
        System.out.println("collecting business subset...");
        while ((line = buffr2.readLine()) != null) {
//            System.out.println(line);
            JSONObject json = (JSONObject) parser.parse(line);
            String catString = (String) json.get("categories");
            if (catString != null){
                String[] busCats = catString.split(",");
                for (String cat : busCats) {
                    cat = cat.trim();
//                    System.out.println(cat);
                    if (top100.containsKey(cat)) {
                        busCount++;
                        pw.write(json.toJSONString());
//                        System.out.println(json.toJSONString());
//                        pw.write(json.toString());
                        pw.println();
                        break;
//                        cats.put(cat, 0); // add this business to output
                    }
//                    cats.put(cat, cats.get(cat) + 1);
//                    docCount++;
                }
            }
        }
        System.out.println("total number of businesses in subset: " + busCount);

        pw.close();
        buffr2.close();
    }
}

