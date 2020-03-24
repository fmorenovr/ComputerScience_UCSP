#ifndef OBJECT_H
#define OBJECT_H

#include "Includes.h"
#include "Node.h"

class Object{
    private:
        int idObject;
        string value;
        int distance;
        bool isParent; //has a node child 
        //Only if the object is parent, has:
        int radius;
        int ptr; //id of the child node
    public:
        Object();
        Object(int,string,int,bool,int,int);
        void calculateEditDistanceWithPivot(Object *);
        int calculateEditDistanceWithOther(Object *);

    friend class Node;
    friend class SlimTree;
};
Object::Object(int pId,string pValue,int pDistance,bool pIsParent,int pRadius, int pPtr){
    this->idObject = pId;
    this->value = pValue;
    this->distance = pDistance;
    this->isParent = pIsParent;
    this->radius = pRadius;
    this->ptr = pPtr;
}
void Object::calculateEditDistanceWithPivot(Object * p){
    const size_t len1 = this->value.size(), len2 = p->value.size();
    vector< vector<unsigned int> > d(len1 + 1, std::vector<unsigned int>(len2 + 1));

    d[0][0] = 0;
    for(unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
    for(unsigned int i = 1; i <= len2; ++i) d[0][i] = i;

    for(unsigned int i = 1; i <= len1; ++i)
        for(unsigned int j = 1; j <= len2; ++j)
            d[i][j] = min(min(d[i - 1][j] + 1, d[i][j - 1] + 1), d[i - 1][j - 1] + (this->value[i - 1] == p->value[j - 1] ? 0 : 1));
    this->distance = d[len1][len2];
}

int Object::calculateEditDistanceWithOther(Object * o){
    const size_t len1 = this->value.size(), len2 = o->value.size();
    vector< vector<unsigned int> > d(len1 + 1, std::vector<unsigned int>(len2 + 1));

    d[0][0] = 0;
    for(unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
    for(unsigned int i = 1; i <= len2; ++i) d[0][i] = i;

    for(unsigned int i = 1; i <= len1; ++i)
        for(unsigned int j = 1; j <= len2; ++j)
            d[i][j] = min(min(d[i - 1][j] + 1, d[i][j - 1] + 1), d[i - 1][j - 1] + (this->value[i - 1] == o->value[j - 1] ? 0 : 1));
    return d[len1][len2];
}
#endif