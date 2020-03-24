#include <fstream>
#include <iostream>
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <chrono>
#include "RTree.h"

using namespace std;
using namespace chrono;
namespace pt = boost::property_tree;

ObjectRTree convertJSONtoObject(string input){
  stringstream fromJSON;
  fromJSON << input;
  pt::ptree iroot;
  pt::read_json(fromJSON, iroot);

  int nivel = iroot.get<int>("order");  
  std::vector<double> minPoint;
  for (pt::ptree::value_type &min : iroot.get_child("minP")){
    minPoint.push_back(min.second.get_value<double>());
  }
  
  std::vector<double> maxPoint;
  for (pt::ptree::value_type &max : iroot.get_child("maxP")){
    maxPoint.push_back(max.second.get_value<double>());
  }

  return ObjectRTree(Rect(minPoint[0], minPoint[1], maxPoint[0], maxPoint[1]), nivel); // xmin, ymin, xmax, ymax
/*  a.nivel_data = nivel;
  a.limits[0] = minPoint[0]; // xmin
  a.limits[1] = maxPoint[0]; // xmax
  a.limits[2] = minPoint[1]; // ymin
  a.limits[3] = maxPoint[1]; // ymax
  return a;*/
}

string convertRegionsToJSON(vector<data_node> data_tree, int size){
  stringstream toJSON;
  //int size = data_tree.size();
  pt::ptree oroot, data, element[size];
  
  for(int i=0; i<size;i++){
    element[i].put<int>("nivel", data_tree[i].nivel_data);
    element[i].put("tag", data_tree[i].tag);
    element[i].put<bool>("leaf", data_tree[i].leaf);
    pt::ptree pointsMax, pointsMin;
    for(int j=0;j<2;j++){
      pt::ptree points;
      pointsMax.put<int>("", data_tree[i].limits[j]);
      pointsMin.put<int>("", data_tree[i].limits[j+2]);
      points.push_back(std::make_pair("", pointsMax));
      points.push_back(std::make_pair("", pointsMin));
      if(j==0){
        element[i].add_child("minP",  points);
      } else if (j==1){
        element[i].add_child("maxP",  points);
      }
    }
    data.push_back(std::make_pair("", element[i]));
  }
  oroot.add_child("data", data);
  
  pt::write_json(toJSON, oroot);
  string output = toJSON.str();
  return output;
}

Rect convertJSONToIDs(string input){
  stringstream fromJSON;
  fromJSON << input;
  pt::ptree iroot;
  pt::read_json(fromJSON, iroot);

  std::vector<double> minPoint;
  for (pt::ptree::value_type &min : iroot.get_child("minP")){
    minPoint.push_back(min.second.get_value<double>());
  }
  
  std::vector<double> maxPoint;
  for (pt::ptree::value_type &max : iroot.get_child("maxP")){
    maxPoint.push_back(max.second.get_value<double>());
  }

  return Rect(minPoint[0], minPoint[1], maxPoint[0], maxPoint[1]);
}

string convertIDsToJSON(vector<int> search_t){
  stringstream toJSON;
  int size = search_t.size();
  pt::ptree oroot, data, element;
  
  for(int i=0; i<size;i++){
    element.put<int>("", search_t[i]);
    data.push_back(std::make_pair("", element));
  }
  oroot.add_child("data", data);
  
  pt::write_json(toJSON, oroot);
  string output = toJSON.str();
  return output;
}

ObjectKNN convertJSONtoKNN(string input){
  stringstream fromJSON;
  fromJSON << input;
  pt::ptree iroot;
  pt::read_json(fromJSON, iroot);
  ValueType x = iroot.get<ValueType>("x");
  ValueType y = iroot.get<ValueType>("y");
  int k = iroot.get<int>("k");
  return ObjectKNN(x,y, k);
}

string convertKNNToJSON(vector<int> search_t, ValueType xpoint, ValueType ypoint){
  stringstream toJSON;
  int size = search_t.size();
  pt::ptree oroot, data, element;
  
  for(int i=0; i<size;i++){
    element.put<int>("", search_t[i]);
    data.push_back(std::make_pair("", element));
  }
  oroot.add_child("data", data);
  oroot.add("x", xpoint);
  oroot.add("y", ypoint);
  
  pt::write_json(toJSON, oroot);
  string output = toJSON.str();
  return output;
}
