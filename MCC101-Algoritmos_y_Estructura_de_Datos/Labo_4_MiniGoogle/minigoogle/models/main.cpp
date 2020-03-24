#include "Includes.h"
#include "SlimTree.h"

int main(){
    Object *o1 = new Object(0,"luis",2,true,4,3);
    Object *o2 = new Object(1,"laasa",2,true,4,3);
    Object *o3 = new Object(2,"looos",2,true,4,3);
    Object *o4 = new Object(3,"lups",2,true,4,3);
    Object *o5 = new Object(4,"jade",2,true,4,3);
    Object *o6 = new Object(5,"jaade",2,true,4,3);
    Object *o7 = new Object(6,"jode",2,true,4,3);
    Object *o8 = new Object(7,"jide",2,true,4,3);
    vector<Object*> objects;
    objects.push_back(o1);
    objects.push_back(o2);
    objects.push_back(o3);
    objects.push_back(o4);
    objects.push_back(o5);
    objects.push_back(o6);
    objects.push_back(o7);
    objects.push_back(o8);
    Node *node = new Node(1,true,objects,o3,8,true);
    node->runSplitting();
    //int dist = o1->calculateEditDistanceWithOther(o2);
    //cout << dist << endl;
}