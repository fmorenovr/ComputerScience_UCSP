#ifndef NODE_H
#define NODE_H

#include "Object.h"
#include "kruskal.h"
typedef  pair<int, int> iPair;
typedef pair<int, iPair> wPair;

class Node{
    private:
        int idNode;
        int * idParent;
        bool isLeaf; //Is leaf or index 
        vector<Object*> objects;
        Object * repObject;
        int C; // number max of entries
        int count;
        bool isFull = false;
        bool isRoot;
    public:
        Node();
        Node(int,bool,vector<Object*>,Object*,int,bool);
        int insertObject(Object * e);
        void runSplitting();
        void incrCount();
    friend class SlimTree;
};
Node::Node(int pId,bool pIsLeaf,vector<Object*> pObjects,Object* pRepObject,int pC,bool pIsRoot){
    count = 0;
    this->idNode = pId;
    this->isLeaf = pIsLeaf;
    this->objects = pObjects;
    this->repObject = pRepObject;
    this->C = pC;
    this->isRoot = pIsRoot;
}
int Node::insertObject(Object * o){
    this->objects.push_back(o);
    return count++;
}
vector< pair<int, iPair> > getParition1(vector< pair<int, iPair> >& partition1,vector< pair<int, iPair> >& mst, int searcher){
    vector< pair<int, iPair> > aux;
    for(auto e : mst){
        if(searcher == e.second.first || searcher == e.second.second)
        {
            if(!(find(partition1.begin(), partition1.end(), e) != partition1.end())){
                partition1.push_back(e);
                aux.push_back(e);
            }

        }
    }
    return aux;
}
void Partition(vector< pair<int, iPair> >& partition1,vector< pair<int, iPair> >& mst, int searcher){
    auto aux = getParition1(partition1,mst,searcher);
    for(auto au : aux){
        Partition(partition1,mst,au.second.first);
        Partition(partition1,mst,au.second.second);
    }
}
void Node::runSplitting(){
    int V = this->objects.size();
    int E = V*(V-1);
    Graph g(V, E);
    for(auto o : this->objects)
    {
        for(auto obj : this->objects)
        {
            if(o->idObject != obj->idObject){
                g.addEdge(o->idObject, obj->idObject, o->calculateEditDistanceWithOther(obj));
            }
        }
    }
    //Build MST
    vector< pair<int, iPair> > MST = g.kruskalMST();

    //Find max edge 
    int max = 0;
    vector< pair<int, iPair> >::iterator i;
    vector< pair<int, iPair> >::iterator maxEdge;
    for(i = MST.begin();i!=MST.end();i++){
        if(max < i->first){
            max = i->first;
            maxEdge = i;
        }
    }
    cout << "*** MST ***" << endl;
    for(auto e : MST)
    {
        cout << e.second.first << " , " << e.second.second << " peso " << e.first << endl;
    }
    cout << " max edge " << endl;
    cout << maxEdge->second.first << " , " << maxEdge->second.second << endl; 
    int u = maxEdge->second.first;
    int v = maxEdge->second.second;
    //delete max edge
    MST.erase(maxEdge);
    vector< pair<int, iPair> > partition1, partition2; 

    Partition(partition1,MST,u);
    cout << " partition1 " << endl;
    for(auto p1 : partition1){
        cout << p1.second.first << " , " << p1.second.second << endl;
    }

    Partition(partition2,MST,v);
        cout << " partition2 " << endl;
    for(auto p2 : partition2){
        cout << p2.second.first << " , " << p2.second.second << endl;
    }
    vector<int> group1, group2;
    max = 0;
    
    for(auto p1 : partition1)
    {
        if(!(find(group1.begin(), group1.end(), p1.second.first) != group1.end()))
            group1.push_back(p1.second.first);
        if(!(find(group1.begin(), group1.end(), p1.second.second) != group1.end()))
            group1.push_back(p1.second.second);
    }
    
    for(auto p2 : partition2)
    {
        if(!(find(group2.begin(), group2.end(), p2.second.first) != group2.end()))
            group2.push_back(p2.second.first);
        if(!(find(group2.begin(), group2.end(), p2.second.second) != group2.end()))
            group2.push_back(p2.second.second);
    }
    //this->idParent->insertObject();

    //chosse pivot group 1
    vector<iPair> maximums;
    for(auto id1 : group1)
    {
        int max = 0;
        for(auto id2 : group1){
            if(id1 != id2){
                if(max < this->objects[id1]->calculateEditDistanceWithOther(this->objects[id2]));
                    max = this->objects[id1]->calculateEditDistanceWithOther(this->objects[id2]);
            }
        }
        maximums.push_back(iPair(id1,max));
    }
    int min = INF;
    int idmin;
    for(auto m : maximums){
        if(min > m.second){
            min = m.second;
            idmin = m.first;
        }
    }
    cout << "pivot 1" << idmin << endl;

    //chosse pivot group 2
vector<iPair> maximums2;
    for(auto idg1 : group2)
    {
        int max = 0;
        for(auto idg2 : group2){
            if(idg1 != idg2){
                if(max < this->objects[idg1]->calculateEditDistanceWithOther(this->objects[idg2]));
                    max = this->objects[idg1]->calculateEditDistanceWithOther(this->objects[idg2]);
            }
        }
        maximums2.push_back(iPair(idg1,max));
    }
    min = INF;
    idmin;
    for(auto mt : maximums2){
        if(min > mt.second){
            min = mt.second;
            idmin = mt.first;
        }
    }
    cout << "pivot 2" << idmin << endl;

}
void Node::incrCount(){
    this->count++;
    if(this->count == this->C) this->isFull = true;
}
#endif