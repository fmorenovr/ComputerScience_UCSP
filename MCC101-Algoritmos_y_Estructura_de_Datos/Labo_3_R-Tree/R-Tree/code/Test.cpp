#include <iostream>
#include "RTree.h"
#include <time.h>   

#define TESTDATA 10000

using namespace std;

typedef double ValueType;

struct Rect rects[TESTDATA];

ValueType a_point[2];
int a_k;

int main()
{
  int dataCount[] = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000};
  srand (time(NULL));
  typedef RTree<ValueType, ValueType, 2, float, 8> MyTree;
  MyTree tree;
  
  for (auto numData : dataCount){
    
    // xmin, ymin, xmax, ymax (for 2 dimensional RTree)
    for(int i=0; i<numData; i++){
        rects[i]=Rect(rand() % 100 + 1.,rand() % 100 + 1.,rand() % 1000 + 1.,rand() % 1000 + 1.);
    }
    
    clock_t t = clock();
    for(int i=0; i<numData; i++)
    {
      tree.Updatetree(rects[i].min, rects[i].max, i); // Note, all values including zero are fine in this version
    }
    t = clock() - t;
    cout << "tomo insertar "<< numData << " elementos : " << ((float)t)/CLOCKS_PER_SEC << endl;

    int nhits;
    Rect rectSearch;
    t = clock();
    for(int i=0; i<numData; i++){
        rectSearch=Rect(rand() % 100 + 1.,rand() % 100 + 1.,rand() % 1000 + 1.,rand() % 1000 + 1.);
        nhits = tree.Search(rectSearch.min, rectSearch.max, MySearchCallback);
    }
    t = clock() - t;
    cout << "tomo buscar Area en "<< numData << " elementos : " << ((float)t)/CLOCKS_PER_SEC << endl;

    t = clock();
    a_k = 100;
    for(int i=0; i<numData; i++){
        a_point[0] = rand() % 200 + 1.;
        a_point[1] = rand() % 800 + 1.;
        tree.Search_knn(a_point, a_k);
    }
    t = clock() - t;
    cout << "tomo buscar los "<< a_k << " KNN en "<< numData << " elementos : " << ((float)t)/CLOCKS_PER_SEC << endl;

    tree.RemoveAll();
  }
  return 0;

}
