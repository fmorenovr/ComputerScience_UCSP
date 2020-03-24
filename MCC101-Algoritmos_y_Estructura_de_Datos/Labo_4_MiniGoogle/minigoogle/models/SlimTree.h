#ifndef SLIMTREE_H
#define SLIMTREE_H

#include "Node.h"

class SlimTree{
    private:
        vector<Node*> node;
        int height;
        double fatFactor;
        Node * root;
    public:
        SlimTree();
        int insertObject(Object * n);
        double getFatFactor();
        double updateFatFactor();
};
SlimTree::SlimTree(){
    root->isLeaf = true;
       
}
int SlimTree::insertObject(Object * obj){
    int minDistance = INF;
    Node * nearestNode;
    Node * minDistanceNode;
    vector<Node*> qualifiesNode;
    if(root->isLeaf)
        if(!root->isFull){
            //fix parameters 
            root->objects.push_back(obj);
            root->incrCount();
        }
        else 
            root->runSplitting();
    else{
        for(auto o : root->objects){
            int distanceToRepO = obj->calculateEditDistanceWithOther(node[o->ptr]->repObject);
            if(node[o->ptr]->repObject->radius <= distanceToRepO){
                if(minDistance > distanceToRepO){
                    minDistance = distanceToRepO;
                    minDistanceNode = node[o->ptr];  
                }
                qualifiesNode.push_back(node[o->ptr]);
            }
            if(minDistance < distanceToRepO){
                nearestNode = node[o->ptr];
                minDistance = distanceToRepO;
            }
        }
        if(qualifiesNode.size() == 1){
            if(!qualifiesNode[0]->isFull){
                qualifiesNode[0]->objects.push_back(obj);
                qualifiesNode[0]->incrCount();
            }
            else
                qualifiesNode[0]->runSplitting();
        }
        else if(qualifiesNode.size() > 0){
            if(!minDistanceNode->isFull){
                minDistanceNode->objects.push_back(obj); 
                minDistanceNode->incrCount();
            }     
            else
                minDistanceNode->runSplitting();
        }
        else{
            if(!nearestNode->isFull){
                nearestNode->objects.push_back(obj);
                nearestNode->incrCount();
            }
            else{
                nearestNode->runSplitting();
            }
        }
    }
}
double SlimTree::getFatFactor(){
}
double SlimTree::updateFatFactor(){

}
#endif