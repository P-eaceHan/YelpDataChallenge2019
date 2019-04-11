/**
 * code to extract and preprocess the Yelp data
 *
 * @author Peace Han
 * @author Krupa Patel
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.Map;

import org.json.simple.JSONObject;
import org.json.simple.JSONArray;
import org.json.simple.parser.*;

public class ExtractJSONtoCSV {

    public static void main(String[] args) {
        JSONParser parser = new JSONParser();
//        File file = new File("/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/yelp_dataset/review.json");
//        File outputfile = new File("/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/IR_FinalProject/output/review.csv");

        File file = new File("../data/output/review_sub.json");
        File outputfile = new File("../data/output/review_sub.csv");

        try {

            System.out.println("Reading JSON file");

            BufferedReader br = new BufferedReader(new FileReader(file));
            PrintWriter pw = new PrintWriter(outputfile);

            String line;

            while ((line = br.readLine()) != null) {
//        		    System.out.println(line);
                JSONObject json = (JSONObject) parser.parse(line);
                StringBuilder sb = new StringBuilder();
//               	    String[] entries = {"business_id", "name", "stars", "review_count","categories"};

                String business_id = (String) json.get("business_id");
                business_id = business_id.replaceAll(",", " ");
                sb.append(business_id);
                sb.append(',');

                String name = (String) json.get("review_id");
                name = name.replaceAll(",", " ");
                sb.append(name);
                sb.append(",");

                String text = (String) json.get("text");
                if (text != null) {
                    text = text.replaceAll("\n", " ");
                    text = text.replaceAll(",", " ");
                    sb.append(text);
                }

                System.out.println("JSON : " + sb.toString());
                pw.write(sb.toString());

                pw.println();
            }
            pw.close();
            br.close();
        } catch (ParseException | IOException e) {
            e.printStackTrace();
        }
    }
}

