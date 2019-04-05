/**
 * code to extract and preprocess the Yelp data
 * @author Peace Han
 * @author Krupa Patel
 */

import java.io.FileReader;
import java.util.Iterator;
import java.util.Map;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.*;

public class ExtractJSON {
    public static void main(String[] args) throws Exception {
        // this string parses just fin
        String json = "{\"business_id\":\"1SWheh84yJXfytovILXOAQ\",\"name\":\"Arizona Biltmore Golf Club\",\"address\":\"2818 E Camino Acequia Drive\",\"city\":\"Phoenix\",\"state\":\"AZ\",\"postal_code\":\"85016\",\"latitude\":33.5221425,\"longitude\":-112.0184807,\"stars\":3.0,\"review_count\":5,\"is_open\":0,\"attributes\":{\"GoodForKids\":\"False\"},\"categories\":\"Golf, Active Life\",\"hours\":null}\n";
        // this string (which is two lines of business.json) throws error
//        String json = "{\"business_id\":\"1SWheh84yJXfytovILXOAQ\",\"name\":\"Arizona Biltmore Golf Club\",\"address\":\"2818 E Camino Acequia Drive\",\"city\":\"Phoenix\",\"state\":\"AZ\",\"postal_code\":\"85016\",\"latitude\":33.5221425,\"longitude\":-112.0184807,\"stars\":3.0,\"review_count\":5,\"is_open\":0,\"attributes\":{\"GoodForKids\":\"False\"},\"categories\":\"Golf, Active Life\",\"hours\":null}\n" +
//                "{\"business_id\":\"QXAEGFB4oINsVuTFxEYKFQ\",\"name\":\"Emerald Chinese Restaurant\",\"address\":\"30 Eglinton Avenue W\",\"city\":\"Mississauga\",\"state\":\"ON\",\"postal_code\":\"L5R 3E7\",\"latitude\":43.6054989743,\"longitude\":-79.652288909,\"stars\":2.5,\"review_count\":128,\"is_open\":1,\"attributes\":{\"RestaurantsReservations\":\"True\",\"GoodForMeal\":\"{'dessert': False, 'latenight': False, 'lunch': True, 'dinner': True, 'brunch': False, 'breakfast': False}\",\"BusinessParking\":\"{'garage': False, 'street': False, 'validated': False, 'lot': True, 'valet': False}\",\"Caters\":\"True\",\"NoiseLevel\":\"u'loud'\",\"RestaurantsTableService\":\"True\",\"RestaurantsTakeOut\":\"True\",\"RestaurantsPriceRange2\":\"2\",\"OutdoorSeating\":\"False\",\"BikeParking\":\"False\",\"Ambience\":\"{'romantic': False, 'intimate': False, 'classy': False, 'hipster': False, 'divey': False, 'touristy': False, 'trendy': False, 'upscale': False, 'casual': True}\",\"HasTV\":\"False\",\"WiFi\":\"u'no'\",\"GoodForKids\":\"True\",\"Alcohol\":\"u'full_bar'\",\"RestaurantsAttire\":\"u'casual'\",\"RestaurantsGoodForGroups\":\"True\",\"RestaurantsDelivery\":\"False\"},\"categories\":\"Specialty Food, Restaurants, Dim Sum, Imported Food, Food, Chinese, Ethnic Food, Seafood\",\"hours\":{\"Monday\":\"9:0-0:0\",\"Tuesday\":\"9:0-0:0\",\"Wednesday\":\"9:0-0:0\",\"Thursday\":\"9:0-0:0\",\"Friday\":\"9:0-1:0\",\"Saturday\":\"9:0-1:0\",\"Sunday\":\"9:0-0:0\"}}\n";
        // the JSONParser object can only handle one JSON object
        // we need to read in the json file and iteratively create a new JSONObject for each line
        Object obj = new JSONParser().parse(json);

        // typecasting obj to JSONObject
        JSONObject jo = (JSONObject) obj;

        // getting firstName and lastName
        String business_id = (String) jo.get("business_id");
        String business_name = (String) jo.get("name");

        System.out.println(business_id);
        System.out.println(business_name);

        /*
        // getting age
        long age = (long) jo.get("age");
        System.out.println(age);

        // getting address
        Map address = ((Map)jo.get("address"));

        // iterating address Map
        Iterator<Map.Entry> itr1 = address.entrySet().iterator();
        while (itr1.hasNext()) {
            Map.Entry pair = itr1.next();
            System.out.println(pair.getKey() + " : " + pair.getValue());
        }

        // getting phoneNumbers
        JSONArray ja = (JSONArray) jo.get("phoneNumbers");

        // iterating phoneNumbers
        Iterator itr2 = ja.iterator();

        while (itr2.hasNext())
        {
            itr1 = ((Map) itr2.next()).entrySet().iterator();
            while (itr1.hasNext()) {
                Map.Entry pair = itr1.next();
                System.out.println(pair.getKey() + " : " + pair.getValue());
            }
        }        */
    }
}

