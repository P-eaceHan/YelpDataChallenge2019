/**
 * code to extract and preprocess the Yelp data
 * @author Peace Han
 * @author Krupa Patel
 */
package Task1;

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

public class ExtractJSONToCSV {
	
    public static void main(String[] args)
    {
           JSONParser parser = new JSONParser();
           // JSON file to be converted to csv
           File file = new File("/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/IR_FinalProject/output/business_sub.json");
           
           // Result output CSV file
           File outputfile = new File("/Volumes/Krupa/MISStudy/Spring 2019/Search/Final Project/IR_FinalProject/output/business_cat.csv");
           
           try {
        	   
        	   System.out.println("Reading JSON file");
        	  
        	   BufferedReader br = new BufferedReader(new FileReader(file));
        	   PrintWriter pw = new PrintWriter(outputfile);

        	   String line;
        	   pw.print("business_ID");
        	   pw.print(",");
//        	   pw.print("review_id");
//        	   pw.print(",");
//        	   pw.print("stars");
//        	   pw.print(",");
//        	   pw.print("cool");
//        	   pw.print(",");
//        	   pw.print("useful");
//        	   pw.print(",");
//        	   pw.print("funny");
//        	   pw.print(",");
//        	   pw.print("text");
        	   pw.print("categories");
        	   pw.println();
        	   
        	   while((line = br.readLine()) != null) 
        	   {
        		    JSONObject json = (JSONObject) parser.parse(line); 
               	    StringBuilder sb = new StringBuilder();
               	    
	   				String business_id = (String) json.get("business_id");
	   				business_id = business_id.replaceAll(",", " ");
	   				business_id = business_id.replaceAll("_","");
	   				business_id = business_id.replaceAll("-","");
	   				sb.append(business_id);
	   				sb.append(',');
	   				
	   				String category = (String) json.get("categories");
	   				category = category.replaceAll(",", " ");
	   				sb.append(category);
//	   				String name = (String) json.get("review_id");
//	   				name = name.replaceAll(",", " ");
//	   				sb.append(name);
//	   				sb.append(",");
//	   				
//	   				String stars = (String) json.get("stars").toString();
//	   				sb.append(stars);
//	   				sb.append(",");
//	   				
//	   				String cool = (String) json.get("cool").toString();
//	   				sb.append(cool);
//	   				sb.append(",");
//	   				
//	   				String useful = (String) json.get("useful").toString();
//	   				sb.append(useful);
//	   				sb.append(",");
//	   			
//	   				String funny = (String) json.get("funny").toString();
//	   				sb.append(funny);
//	   				sb.append(",");
//	   				
//	   				String text = (String) json.get("text");
//	   				if(text != null) {
//	   					text = text.replaceAll("\n"," ");
//	   					text = text.replaceAll(",", " ");
//	   					text = text.replaceAll("\"", " ");
//	   					text = text.replaceAll("\r", " ");
//	   					text = text.replaceAll("AND", "and");
//	   					text = text.replaceAll("NOT", "not");
//	   					text = text.replaceAll("OR", "or");
//	   					sb.append(text);
//	   				}
	   	
        		    System.out.println("JSON : " + sb.toString());
	   				pw.write(sb.toString());
	   				   				
	   				pw.println();  	   				
        		   }
   			pw.close();
   			br.close();
           }
           catch(ParseException | IOException e) {
        	   e.printStackTrace();
           }
    }
}

