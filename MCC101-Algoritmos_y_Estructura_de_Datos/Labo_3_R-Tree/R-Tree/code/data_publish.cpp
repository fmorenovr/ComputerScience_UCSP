#include "paho_client.h"

// g++ data_publish.cpp -lpaho-mqttpp3

int main(int argc, char* argv[]) {
  string payload;

  auto tm = steady_clock::now();
  do{
  this_thread::sleep_until(tm);
  srand((unsigned)time(0));
  vector<data_node> data_tree;
  
  data_tree.push_back({{(rand()%800)+1.0,(rand()%800)+1.0,(rand()%600)+1.0,(rand()%600)+1.0}, false, 1, "R0"});
  data_tree.push_back({{(rand()%800)+1.0,(rand()%800)+1.0,(rand()%600)+1.0,(rand()%600)+1.0}, false, 2, "R1"});
  data_tree.push_back({{(rand()%800)+1.0,(rand()%800)+1.0,(rand()%600)+1.0,(rand()%600)+1.0}, false, 2, "R2"});
  
  payload = convertRegionsToJSON(data_tree, data_tree.size());
  tm += PERIOD;
  
  ObjectRTree a = convertJSONtoObject("{\"order\":1, \"minP\":[6, 3], \"maxP\":[5, 9]}");
  
  cout << a.order << " " << a.rect.min[0] << " " << a.rect.min[1] << " " << a.rect.max[0] << " " << a.rect.max[1] << endl;
  
  Rect search = convertJSONToIDs("{\"minP\":[6, 3], \"maxP\":[5, 9]}");
  
  cout << search.min[0] << " " << search.min[1] << " " << search.max[0] << " " << search.max[1] << endl;
  
  vector<int> ss = {0,1,2,3,6,7};
  
  string data = convertIDsToJSON(ss);
  
  cout << data << endl;
  
  //}while( mqttPublish("cpp/insert", payload) == 0);
  }while(1);
}
