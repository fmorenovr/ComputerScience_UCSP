#ifndef RTREE_H
#define RTREE_H

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>

#include <algorithm>
#include <functional>
#include <vector>
using namespace std;

#define ASSERT assert
#ifndef Min
  #define Min std::min
#endif //Min
#ifndef Max
  #define Max std::max
#endif //Max

//
// RTree.h
//

#define RTREE_TEMPLATE template<class DATATYPE, class ELEMTYPE, int NUMDIMS, class ELEMTYPEREAL, int TMAXNODES, int TMINNODES>
#define RTREE_QUAL RTree<DATATYPE, ELEMTYPE, NUMDIMS, ELEMTYPEREAL, TMAXNODES, TMINNODES>

#define RTREE_DONT_USE_MEMPOOLS
#define RTREE_USE_SPHERICAL_VOLUME

typedef double ValueType;

struct Rect{
  Rect()  {}

  Rect(ValueType a_minX, ValueType a_minY, ValueType a_maxX, ValueType a_maxY)
  {
    min[0] = a_minX;
    min[1] = a_minY;

    max[0] = a_maxX;
    max[1] = a_maxY;
  }

  ValueType min[2];
  ValueType max[2];
};

struct ObjectRTree {
  ObjectRTree(Rect rectObj, int ordered){
    rect = rectObj;
    order = ordered;
  }
  Rect rect;
  int order;
};

struct ObjectKNN{
  ObjectKNN(ValueType xpoint, ValueType ypoint, int kObj){
    points[0] = xpoint;
    points[1] = ypoint;
    k = kObj;
  }
  ValueType points[2];
  int k;
};

struct  data_node{
  double limits[4];
  bool leaf;
  int nivel_data; // nivel
  string tag; // R1
};
vector<data_node> data_tree;
vector<int> search_export;
vector<int> search_knn_export;
int export_aux;

bool MySearchCallback(ValueType id){
  //cout << "Hit data rect " << id << "\n";
  return true; // keep going
}

template<class DATATYPE, class ELEMTYPE, int NUMDIMS, class ELEMTYPEREAL = ELEMTYPE, int TMAXNODES=4, int TMINNODES = TMAXNODES/2>
class RTree
{
protected:

  struct Node;  // Fwd decl.  Used by other internal structs and iterator

public:

  enum
  {
    MAXNODES = TMAXNODES,                         ///< Max elements in node
    MINNODES = TMINNODES,                         ///< Min elements in node
  };

public:

  RTree();
  RTree(const RTree& other);
  virtual ~RTree();

  void get_tags();
  void Insert(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], const DATATYPE& a_dataId);
  void Updatetree(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], const DATATYPE& a_dataId);
  void Remove(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], const DATATYPE& a_dataId);
  int Search(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], std::function<bool (const DATATYPE&)> callback) const;
  void Search_knn(const ELEMTYPE a_point[NUMDIMS], int a_k);
  int Search_1(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], std::function<bool (const DATATYPE&)> callback) const;
  void RemoveAll();
  int Count();

  class Iterator{
  private:

    enum { MAX_STACK = 32 };

    struct StackElement{
      Node* m_node;
      int m_branchIndex;
    };

  public:

    Iterator()                                    { Init(); }

    ~Iterator()                                   { }

    bool IsNull()                                 { return (m_tos <= 0); }

    bool IsNotNull()                              { return (m_tos > 0); }

    DATATYPE& operator*()
    {
      ASSERT(IsNotNull());
      StackElement& curTos = m_stack[m_tos - 1];
      return curTos.m_node->m_branch[curTos.m_branchIndex].m_data;
    }

    const DATATYPE& operator*() const
    {
      ASSERT(IsNotNull());
      StackElement& curTos = m_stack[m_tos - 1];
      return curTos.m_node->m_branch[curTos.m_branchIndex].m_data;
    }

    bool operator++()                             { return FindNextData(); }

    void GetBounds(ELEMTYPE a_min[NUMDIMS], ELEMTYPE a_max[NUMDIMS])
    {
      ASSERT(IsNotNull());
      StackElement& curTos = m_stack[m_tos - 1];
      Branch& curBranch = curTos.m_node->m_branch[curTos.m_branchIndex];

      for(int index = 0; index < NUMDIMS; ++index)
      {
        a_min[index] = curBranch.m_rect.m_min[index];
        a_max[index] = curBranch.m_rect.m_max[index];
      }
    }

  private:

    void Init()                                   { m_tos = 0; }

    bool FindNextData()
    {
      for(;;)
      {
        if(m_tos <= 0)
        {
          return false;
        }
        StackElement curTos = Pop();

        if(curTos.m_node->IsLeaf())
        {
          if(curTos.m_branchIndex+1 < curTos.m_node->m_count)
          {
            Push(curTos.m_node, curTos.m_branchIndex + 1);
            return true;
          }
        }
        else
        {
          if(curTos.m_branchIndex+1 < curTos.m_node->m_count)
          {
            Push(curTos.m_node, curTos.m_branchIndex + 1);
          }
          Node* nextLevelnode = curTos.m_node->m_branch[curTos.m_branchIndex].m_child;
          Push(nextLevelnode, 0);

          if(nextLevelnode->IsLeaf())
          {
            return true;
          }
        }
      }
    }

    /// Push node and branch onto iteration stack (For internal use only)
    void Push(Node* a_node, int a_branchIndex)
    {
      m_stack[m_tos].m_node = a_node;
      m_stack[m_tos].m_branchIndex = a_branchIndex;
      ++m_tos;
      ASSERT(m_tos <= MAX_STACK);
    }

    /// Pop element off iteration stack (For internal use only)
    StackElement& Pop()
    {
      ASSERT(m_tos > 0);
      --m_tos;
      return m_stack[m_tos];
    }

    StackElement m_stack[MAX_STACK]; 
    int m_tos;                                 

    friend class RTree; 
  };

  void GetFirst(Iterator& a_it)
  {
    a_it.Init();
    Node* first = m_root;
    while(first)
    {
      if(first->IsInternalNode() && first->m_count > 1)
      {
        a_it.Push(first, 1); // Descend sibling branch later
      }
      else if(first->IsLeaf())
      {
        if(first->m_count)
        {
          a_it.Push(first, 0);
        }
        break;
      }
      first = first->m_branch[0].m_child;
    }
  }

  void GetNext(Iterator& a_it)                    { ++a_it; }

  bool IsNull(Iterator& a_it)                     { return a_it.IsNull(); }

  DATATYPE& GetAt(Iterator& a_it)                 { return *a_it; }

protected:

  struct Rect
  {
    ELEMTYPE m_min[NUMDIMS];      
    ELEMTYPE m_max[NUMDIMS];      
  };

  struct Branch
  {
    Rect m_rect;                  
    Node* m_child;                
    int m_data;                   
  };

  struct Node
  {
    bool IsInternalNode(){ return (m_level > 0); }
    bool IsLeaf(){ return (m_level == 0); }

    int m_count;                    
    int m_level;                    
    Branch m_branch[MAXNODES];      
  };

  struct ListNode
  {
    ListNode* m_next;               
    Node* m_node;                   
  };

  struct NearRect{
    double m_rectDistance = 99999;  
    Rect m_rect;   
    DATATYPE m_data; 
  };
   struct BranchDist{
    int m_index; 
    double m_branchDistance; 
    Branch m_branch; 
  };

  struct PartitionVars
  {
    enum { NOT_TAKEN = -1 }; 

    int m_partition[MAXNODES+1];
    int m_total;
    int m_minFill;
    int m_count[2];
    Rect m_cover[2];
    ELEMTYPEREAL m_area[2];

    Branch m_branchBuf[MAXNODES+1];
    int m_branchCount;
    Rect m_coverSplit;
    ELEMTYPEREAL m_coverSplitArea;
  };

  Node* AllocNode();
  void FreeNode(Node* a_node);
  void InitNode(Node* a_node);
  void InitRect(Rect* a_rect);
  bool InsertRectRec(const Branch& a_branch, Node* a_node, Node** a_newNode, int a_level);
  bool InsertRect(const Branch& a_branch, Node** a_root, int a_level);
  Rect NodeCover(Node* a_node);
  bool AddBranch(const Branch* a_branch, Node* a_node, Node** a_newNode);
  void DisconnectBranch(Node* a_node, int a_index);
  int PickBranch(const Rect* a_rect, Node* a_node);
  Rect CombineRect(const Rect* a_rectA, const Rect* a_rectB);
  void SplitNode(Node* a_node, const Branch* a_branch, Node** a_newNode);
  ELEMTYPEREAL RectSphericalVolume(Rect* a_rect);
  ELEMTYPEREAL RectVolume(Rect* a_rect);
  ELEMTYPEREAL CalcRectVolume(Rect* a_rect);
  void GetBranches(Node* a_node, const Branch* a_branch, PartitionVars* a_parVars);
  void ChoosePartition(PartitionVars* a_parVars, int a_minFill);
  void LoadNodes(Node* a_nodeA, Node* a_nodeB, PartitionVars* a_parVars);
  void InitParVars(PartitionVars* a_parVars, int a_maxRects, int a_minFill);
  void PickSeeds(PartitionVars* a_parVars);
  void Classify(int a_index, int a_group, PartitionVars* a_parVars);
  bool RemoveRect(Rect* a_rect, const DATATYPE& a_id, Node** a_root);
  bool RemoveRectRec(Rect* a_rect, const DATATYPE& a_id, Node* a_node, ListNode** a_listNode);
  ListNode* AllocListNode();
  void FreeListNode(ListNode* a_listNode);
  bool Overlap(Rect* a_rectA, Rect* a_rectB) const;
  bool Cover(Rect* a_rectA, Branch* a_rectB) const;
  bool Cover_1(Rect* a_rectA, Rect* a_rectB) const;
  void ReInsert(Node* a_node, ListNode** a_listNode);
  bool Search(Node* a_node, Rect* a_rect, int& a_foundCount, std::function<bool (const DATATYPE&)> callback) const;
  bool Search_1(Node* a_node, Rect* a_rect, int& a_foundCount, std::function<bool (const DATATYPE&)> callback) const;
  void RemoveAllRec(Node* a_node);
  void Reset();
  void CountRec(Node* a_node, int& a_count);
  void read_MBR_tree(Node *p_node);
  void Search_nn(const ELEMTYPE* a_point, Node* a_node, NearRect* a_nearRects, int &k);
  double ComputeDistance(const ELEMTYPE* a_point, const Rect* a_rect);
  void SortBranchDistList(BranchDist a_branchDistList[], int a_length);
  void AddNearNeighbor(NearRect a_nearRects[], NearRect& a_nearRect, int &k);

  Node* m_root;                                  
  ELEMTYPEREAL m_unitSphereVolume;               
};

RTREE_TEMPLATE
RTREE_QUAL::RTree()
{
  ASSERT(MAXNODES > MINNODES);
  ASSERT(MINNODES > 0);

  const float UNIT_SPHERE_VOLUMES[] = {
    0.000000f, 2.000000f, 3.141593f, 
    4.188790f, 4.934802f, 5.263789f, 
    5.167713f, 4.724766f, 4.058712f, 
    3.298509f, 2.550164f, 1.884104f, 
    1.335263f, 0.910629f, 0.599265f, 
    0.381443f, 0.235331f, 0.140981f, 
    0.082146f, 0.046622f, 0.025807f, 
  };

  m_root = AllocNode();
  m_root->m_level = 0;
  m_unitSphereVolume = (ELEMTYPEREAL)UNIT_SPHERE_VOLUMES[NUMDIMS];
}


RTREE_TEMPLATE
RTREE_QUAL::RTree(const RTree& other) : RTree()
{
	CopyRec(m_root, other.m_root);
}


RTREE_TEMPLATE
RTREE_QUAL::~RTree()
{
  Reset(); // Free, or reset node memory
}


RTREE_TEMPLATE
void RTREE_QUAL::Insert(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], const DATATYPE& a_dataId)
{
#ifdef _DEBUG
  for(int index=0; index<NUMDIMS; ++index)
  {
    ASSERT(a_min[index] <= a_max[index]);
  }
#endif //_DEBUG

  Branch branch;
  branch.m_data = a_dataId;
  branch.m_child = NULL;

  for(int axis=0; axis<NUMDIMS; ++axis)
  {
    branch.m_rect.m_min[axis] = a_min[axis];
    branch.m_rect.m_max[axis] = a_max[axis];
  }

  InsertRect(branch, &m_root, 0);
}

RTREE_TEMPLATE
void RTREE_QUAL::read_MBR_tree(Node *p_node){
  ASSERT(p_node);
  ASSERT(p_node->m_level >= 0);
  //cout<<"EXPORT_AUX= "<<export_aux<<", level = "<<p_node->m_level<<endl;
  if (export_aux==-1) {
    //cout<<"GENERANDO MBR NODO RAIZ"<<endl;
    //PLOTEO DE NODO RAIZ
    export_aux++;
    data_tree.push_back(data_node());
    data_tree[export_aux].leaf=false;
    data_tree[export_aux].nivel_data=p_node->m_level+1;
    for(int axis=0; axis<NUMDIMS; ++axis)
    {
      data_tree[0].limits[2*axis]=p_node->m_branch[0].m_rect.m_min[axis];
      data_tree[0].limits[2*axis+1]=p_node->m_branch[0].m_rect.m_max[axis];
    }
    for(int index=1; index < p_node->m_count; index++)
    {
      for(int axis=0; axis<NUMDIMS; ++axis)
      {
        if (data_tree[export_aux].limits[2*axis]>p_node->m_branch[index].m_rect.m_min[axis]) {
          data_tree[export_aux].limits[2*axis]=p_node->m_branch[index].m_rect.m_min[axis];
        }
        if (data_tree[export_aux].limits[2*axis+1]<p_node->m_branch[index].m_rect.m_max[axis]) {
          data_tree[export_aux].limits[2*axis+1]=p_node->m_branch[index].m_rect.m_max[axis];
        }
      }
    }
    read_MBR_tree(p_node);
  }
  else{
    if (p_node->m_level > 0) {

      for(int index=0; index < p_node->m_count; index++)
      {
        //cout<<"Acceso a nivel: "<<p_node->m_level<<", NE: "<<p_node->m_count<<endl;
        //cout<<"--------------------------------"<<endl;
        //cout<<"minx,miny,maxx,maxy: [ ";
        export_aux++;
        data_tree.push_back(data_node());
        for(int axis=0; axis<NUMDIMS; ++axis)
        {
          data_tree[export_aux].limits[2*axis]=p_node->m_branch[index].m_rect.m_min[axis];
          //cout<<data_tree[export_aux].limits[2*axis]<<" ";

          data_tree[export_aux].limits[2*axis+1]=p_node->m_branch[index].m_rect.m_max[axis];
          //cout<<data_tree[export_aux].limits[2*axis+1]<<" ";
        }
        //data_tree[export_aux].tag="R"+to_string(p_node->m_level)+"_"+to_string(index);
        //cout<<"]"<<endl;

        data_tree[export_aux].leaf=false;
        data_tree[export_aux].nivel_data=p_node->m_level;
        //cout<<"export_aux_BRANCH = "<<export_aux<<endl;
        read_MBR_tree(p_node->m_branch[index].m_child);
      }
    }
  }
}


RTREE_TEMPLATE
void RTREE_QUAL::get_tags(){
   int tag_aux=0;
  //cout<<"M_ROOT LEVEL"<<m_root->m_level<<endl;
  for (int i = m_root->m_level; i >-1; i--) {
    //cout<<"level: "<<i<<endl;
    for (int j = 0; j < export_aux+1; j++) {
      //cout<<"DATA_TREE LEVEL: "<<data_tree[j].nivel_data<<", j = "<<i<<endl;
      if (data_tree[j].nivel_data==i) {
        //cout<<"Tag_aux: "<<tag_aux<<", J= "<<j<<endl;
        data_tree[j].tag="R"+to_string(tag_aux);
        tag_aux++;
      }
    }
  }
}

RTREE_TEMPLATE
void RTREE_QUAL::Updatetree(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], const DATATYPE& a_dataId)
{
#ifdef _DEBUG
  for(int index=0; index<NUMDIMS; ++index)
  {
    ASSERT(a_min[index] <= a_max[index]);
  }
#endif //_DEBUG

  export_aux=-1;
  //cout<<"Insertando rectangulo "<<a_dataId+1<<" en arbol."<<endl;
  Insert(a_min, a_max, a_dataId);
  //cout<<"Ploteando elementos:"<<endl;
  //cout<<"--------------------"<<endl;
  //cout<<"--------------------"<<endl;
  //cout<<"--------------------"<<endl;
  //cout<<"Elementos en root: "<<m_root->m_count<<endl;
  read_MBR_tree(m_root);
  get_tags();
  //cout<<"PLOTING EXPORT DATA"<<endl;
  //cout<<"N_MBR: "<<export_aux+1<<endl;
  //cout<<"-----------------------------------------"<<endl;
  //cout<<"-----------------------------------------"<<endl;
  //cout<<"-----------------------------------------"<<endl;
  /*for (int i = 0; i < export_aux+1; i++) {
    //cout<<"MBR = [ ";
    for (int j = 0; j < 4; j++) {
      cout << data_tree[i].limits[j] << " ";
    }
    //cout<<" ]"<<endl;
    //cout<<"Leaf: "<<data_tree[i].leaf<<endl;
    //cout<<"ID: "<<data_tree[i].nivel_data<<endl;
    //cout<<"Tag: "<<data_tree[i].tag<<endl;
    //cout<<"----------------------------------------"<<endl;
    //cout<<"----------------------------------------"<<endl;
    //cout<<"----------------------------------------"<<endl;
  }*/
}

RTREE_TEMPLATE
void RTREE_QUAL::Remove(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], const DATATYPE& a_dataId)
{
#ifdef _DEBUG
  for(int index=0; index<NUMDIMS; ++index)
  {
    ASSERT(a_min[index] <= a_max[index]);
  }
#endif //_DEBUG

  Rect rect;

  for(int axis=0; axis<NUMDIMS; ++axis)
  {
    rect.m_min[axis] = a_min[axis];
    rect.m_max[axis] = a_max[axis];
  }

  RemoveRect(&rect, a_dataId, &m_root);
}


RTREE_TEMPLATE
int RTREE_QUAL::Search_1(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], std::function<bool (const DATATYPE&)> callback) const
{
#ifdef _DEBUG
  for(int index=0; index<NUMDIMS; ++index)
  {
    ASSERT(a_min[index] <= a_max[index]);
  }
#endif //_DEBUG

  Rect rect;

  for(int axis=0; axis<NUMDIMS; ++axis)
  {
    rect.m_min[axis] = a_min[axis];
    rect.m_max[axis] = a_max[axis];
  }

  // NOTE: May want to return search result another way, perhaps returning the number of found elements here.

  int foundCount = 0;
  Search(m_root, &rect, foundCount, callback);

  return foundCount;
}

RTREE_TEMPLATE
int RTREE_QUAL::Search(const ELEMTYPE a_min[NUMDIMS], const ELEMTYPE a_max[NUMDIMS], std::function<bool (const DATATYPE&)> callback) const
{
#ifdef _DEBUG
  for(int index=0; index<NUMDIMS; ++index)
  {
    ASSERT(a_min[index] <= a_max[index]);
  }
#endif //_DEBUG

  Rect rect;
  search_export.clear();

  for(int axis=0; axis<NUMDIMS; ++axis)
  {
    rect.m_min[axis] = a_min[axis];
    rect.m_max[axis] = a_max[axis];
  }

  // NOTE: May want to return search result another way, perhaps returning the number of found elements here.

  int foundCount = 0;
  Search(m_root, &rect, foundCount, callback);
  //cout<<"Imprimiendo vector search: "<<endl;
  //cout<<"---------------------------"<<endl;
  /*for (int i = 0; i < search_export.size(); i++) {
    cout<<"Elemento "<<i<<" con ID "<<search_export[i]<<endl;
  }*/

  return foundCount;
}


RTREE_TEMPLATE
int RTREE_QUAL::Count()
{
  int count = 0;
  CountRec(m_root, count);

  return count;
}



RTREE_TEMPLATE
void RTREE_QUAL::CountRec(Node* a_node, int& a_count)
{
  if(a_node->IsInternalNode())  // not a leaf node
  {
    for(int index = 0; index < a_node->m_count; ++index)
    {
      CountRec(a_node->m_branch[index].m_child, a_count);
    }
  }
  else // A leaf node
  {
    a_count += a_node->m_count;
  }
}

RTREE_TEMPLATE
void RTREE_QUAL::RemoveAll()
{
  // Delete all existing nodes
  Reset();

  m_root = AllocNode();
  m_root->m_level = 0;
}


RTREE_TEMPLATE
void RTREE_QUAL::Reset()
{
#ifdef RTREE_DONT_USE_MEMPOOLS
  RemoveAllRec(m_root);
#else 
#endif 
}


RTREE_TEMPLATE
void RTREE_QUAL::RemoveAllRec(Node* a_node)
{
  ASSERT(a_node);
  ASSERT(a_node->m_level >= 0);

  if(a_node->IsInternalNode()) // This is an internal node in the tree
  {
    for(int index=0; index < a_node->m_count; ++index)
    {
      RemoveAllRec(a_node->m_branch[index].m_child);
    }
  }
  FreeNode(a_node);
}


RTREE_TEMPLATE
typename RTREE_QUAL::Node* RTREE_QUAL::AllocNode()
{
  Node* newNode;
#ifdef RTREE_DONT_USE_MEMPOOLS
  newNode = new Node;
#else // RTREE_DONT_USE_MEMPOOLS
  // EXAMPLE
#endif // RTREE_DONT_USE_MEMPOOLS
  InitNode(newNode);
  return newNode;
}


RTREE_TEMPLATE
void RTREE_QUAL::FreeNode(Node* a_node)
{
  ASSERT(a_node);

#ifdef RTREE_DONT_USE_MEMPOOLS
  delete a_node;
#else // RTREE_DONT_USE_MEMPOOLS
  // EXAMPLE
#endif // RTREE_DONT_USE_MEMPOOLS
}

RTREE_TEMPLATE
typename RTREE_QUAL::ListNode* RTREE_QUAL::AllocListNode()
{
#ifdef RTREE_DONT_USE_MEMPOOLS
  return new ListNode;
#else // RTREE_DONT_USE_MEMPOOLS
  // EXAMPLE
#endif // RTREE_DONT_USE_MEMPOOLS
}


RTREE_TEMPLATE
void RTREE_QUAL::FreeListNode(ListNode* a_listNode)
{
#ifdef RTREE_DONT_USE_MEMPOOLS
  delete a_listNode;
#else // RTREE_DONT_USE_MEMPOOLS
  // EXAMPLE
#endif // RTREE_DONT_USE_MEMPOOLS
}


RTREE_TEMPLATE
void RTREE_QUAL::InitNode(Node* a_node)
{
  a_node->m_count = 0;
  a_node->m_level = -1;
}


RTREE_TEMPLATE
void RTREE_QUAL::InitRect(Rect* a_rect)
{
  for(int index = 0; index < NUMDIMS; ++index)
  {
    a_rect->m_min[index] = (ELEMTYPE)0;
    a_rect->m_max[index] = (ELEMTYPE)0;
  }
}

RTREE_TEMPLATE
bool RTREE_QUAL::InsertRectRec(const Branch& a_branch, Node* a_node, Node** a_newNode, int a_level)
{
  ASSERT(a_node && a_newNode);
  ASSERT(a_level >= 0 && a_level <= a_node->m_level);
  if(a_node->m_level > a_level)
  {
    Node* otherNode;

    int index = PickBranch(&a_branch.m_rect, a_node);

    bool childWasSplit = InsertRectRec(a_branch, a_node->m_branch[index].m_child, &otherNode, a_level);

    if (!childWasSplit)
    {
      a_node->m_branch[index].m_rect = CombineRect(&a_branch.m_rect, &(a_node->m_branch[index].m_rect));
      return false;
    }
    else
    {
      a_node->m_branch[index].m_rect = NodeCover(a_node->m_branch[index].m_child);
      Branch branch;
      branch.m_child = otherNode;
      branch.m_rect = NodeCover(otherNode);

      return AddBranch(&branch, a_node, a_newNode);
    }
  }
  else if(a_node->m_level == a_level)
  {
    return AddBranch(&a_branch, a_node, a_newNode);
  }
  else
  {
    ASSERT(0);
    return false;
  }
}

RTREE_TEMPLATE
bool RTREE_QUAL::InsertRect(const Branch& a_branch, Node** a_root, int a_level)
{
  ASSERT(a_root);
  ASSERT(a_level >= 0 && a_level <= (*a_root)->m_level);
#ifdef _DEBUG
  for(int index=0; index < NUMDIMS; ++index)
  {
    ASSERT(a_branch.m_rect.m_min[index] <= a_branch.m_rect.m_max[index]);
  }
#endif //_DEBUG

  Node* newNode;

  if(InsertRectRec(a_branch, *a_root, &newNode, a_level))  // Root split
  {
    Node* newRoot = AllocNode();
    newRoot->m_level = (*a_root)->m_level + 1;

    Branch branch;

    branch.m_rect = NodeCover(*a_root);
    branch.m_child = *a_root;
    AddBranch(&branch, newRoot, NULL);

    branch.m_rect = NodeCover(newNode);
    branch.m_child = newNode;
    AddBranch(&branch, newRoot, NULL);

    *a_root = newRoot;

    return true;
  }

  return false;
}

RTREE_TEMPLATE
typename RTREE_QUAL::Rect RTREE_QUAL::NodeCover(Node* a_node)
{
  ASSERT(a_node);

  Rect rect = a_node->m_branch[0].m_rect;
  for(int index = 1; index < a_node->m_count; ++index)
  {
     rect = CombineRect(&rect, &(a_node->m_branch[index].m_rect));
  }

  return rect;
}

RTREE_TEMPLATE
bool RTREE_QUAL::AddBranch(const Branch* a_branch, Node* a_node, Node** a_newNode)
{
  ASSERT(a_branch);
  ASSERT(a_node);

  if(a_node->m_count < MAXNODES)  // Split won't be necessary
  {
    a_node->m_branch[a_node->m_count] = *a_branch;
    ++a_node->m_count;

    return false;
  }
  else
  {
    ASSERT(a_newNode);

    SplitNode(a_node, a_branch, a_newNode);
    return true;
  }
}


RTREE_TEMPLATE
void RTREE_QUAL::DisconnectBranch(Node* a_node, int a_index)
{
  ASSERT(a_node && (a_index >= 0) && (a_index < MAXNODES));
  ASSERT(a_node->m_count > 0);

  a_node->m_branch[a_index] = a_node->m_branch[a_node->m_count - 1];

  --a_node->m_count;
}

RTREE_TEMPLATE
int RTREE_QUAL::PickBranch(const Rect* a_rect, Node* a_node)
{
  ASSERT(a_rect && a_node);

  bool firstTime = true;
  ELEMTYPEREAL increase;
  ELEMTYPEREAL bestIncr = (ELEMTYPEREAL)-1;
  ELEMTYPEREAL area;
  ELEMTYPEREAL bestArea;
  int best = 0;
  Rect tempRect;

  for(int index=0; index < a_node->m_count; ++index)
  {
    Rect* curRect = &a_node->m_branch[index].m_rect;
    area = CalcRectVolume(curRect);
    tempRect = CombineRect(a_rect, curRect);
    increase = CalcRectVolume(&tempRect) - area;
    if((increase < bestIncr) || firstTime)
    {
      best = index;
      bestArea = area;
      bestIncr = increase;
      firstTime = false;
    }
    else if((increase == bestIncr) && (area < bestArea))
    {
      best = index;
      bestArea = area;
      bestIncr = increase;
    }
  }
  return best;
}

RTREE_TEMPLATE
typename RTREE_QUAL::Rect RTREE_QUAL::CombineRect(const Rect* a_rectA, const Rect* a_rectB)
{
  ASSERT(a_rectA && a_rectB);

  Rect newRect;

  for(int index = 0; index < NUMDIMS; ++index)
  {
    newRect.m_min[index] = Min(a_rectA->m_min[index], a_rectB->m_min[index]);
    newRect.m_max[index] = Max(a_rectA->m_max[index], a_rectB->m_max[index]);
  }

  return newRect;
}

RTREE_TEMPLATE
void RTREE_QUAL::SplitNode(Node* a_node, const Branch* a_branch, Node** a_newNode)
{
  ASSERT(a_node);
  ASSERT(a_branch);

  PartitionVars localVars;
  PartitionVars* parVars = &localVars;

  GetBranches(a_node, a_branch, parVars);
  ChoosePartition(parVars, MINNODES);

  *a_newNode = AllocNode();
  (*a_newNode)->m_level = a_node->m_level;

  a_node->m_count = 0;
  LoadNodes(a_node, *a_newNode, parVars);

  ASSERT((a_node->m_count + (*a_newNode)->m_count) == parVars->m_total);
}


RTREE_TEMPLATE
ELEMTYPEREAL RTREE_QUAL::RectVolume(Rect* a_rect)
{
  ASSERT(a_rect);

  ELEMTYPEREAL volume = (ELEMTYPEREAL)1;

  for(int index=0; index<NUMDIMS; ++index)
  {
    volume *= a_rect->m_max[index] - a_rect->m_min[index];
  }

  ASSERT(volume >= (ELEMTYPEREAL)0);

  return volume;
}


RTREE_TEMPLATE
ELEMTYPEREAL RTREE_QUAL::RectSphericalVolume(Rect* a_rect)
{
  ASSERT(a_rect);

  ELEMTYPEREAL sumOfSquares = (ELEMTYPEREAL)0;
  ELEMTYPEREAL radius;

  for(int index=0; index < NUMDIMS; ++index)
  {
    ELEMTYPEREAL halfExtent = ((ELEMTYPEREAL)a_rect->m_max[index] - (ELEMTYPEREAL)a_rect->m_min[index]) * 0.5f;
    sumOfSquares += halfExtent * halfExtent;
  }

  radius = (ELEMTYPEREAL)sqrt(sumOfSquares);

  if(NUMDIMS == 3)
  {
    return (radius * radius * radius * m_unitSphereVolume);
  }
  else if(NUMDIMS == 2)
  {
    return (radius * radius * m_unitSphereVolume);
  }
  else
  {
    return (ELEMTYPEREAL)(pow(radius, NUMDIMS) * m_unitSphereVolume);
  }
}


RTREE_TEMPLATE
ELEMTYPEREAL RTREE_QUAL::CalcRectVolume(Rect* a_rect)
{
#ifdef RTREE_USE_SPHERICAL_VOLUME
  return RectSphericalVolume(a_rect); // Slower but helps certain merge cases
#else // RTREE_USE_SPHERICAL_VOLUME
  return RectVolume(a_rect); // Faster but can cause poor merges
#endif // RTREE_USE_SPHERICAL_VOLUME
}


RTREE_TEMPLATE
void RTREE_QUAL::GetBranches(Node* a_node, const Branch* a_branch, PartitionVars* a_parVars)
{
  ASSERT(a_node);
  ASSERT(a_branch);

  ASSERT(a_node->m_count == MAXNODES);

  // Load the branch buffer
  for(int index=0; index < MAXNODES; ++index)
  {
    a_parVars->m_branchBuf[index] = a_node->m_branch[index];
  }
  a_parVars->m_branchBuf[MAXNODES] = *a_branch;
  a_parVars->m_branchCount = MAXNODES + 1;

  // Calculate rect containing all in the set
  a_parVars->m_coverSplit = a_parVars->m_branchBuf[0].m_rect;
  for(int index=1; index < MAXNODES+1; ++index)
  {
    a_parVars->m_coverSplit = CombineRect(&a_parVars->m_coverSplit, &a_parVars->m_branchBuf[index].m_rect);
  }
  a_parVars->m_coverSplitArea = CalcRectVolume(&a_parVars->m_coverSplit);
}

RTREE_TEMPLATE
void RTREE_QUAL::ChoosePartition(PartitionVars* a_parVars, int a_minFill)
{
  ASSERT(a_parVars);

  ELEMTYPEREAL biggestDiff;
  int group, chosen = 0, betterGroup = 0;

  InitParVars(a_parVars, a_parVars->m_branchCount, a_minFill);
  PickSeeds(a_parVars);

  while (((a_parVars->m_count[0] + a_parVars->m_count[1]) < a_parVars->m_total)
       && (a_parVars->m_count[0] < (a_parVars->m_total - a_parVars->m_minFill))
       && (a_parVars->m_count[1] < (a_parVars->m_total - a_parVars->m_minFill)))
  {
    biggestDiff = (ELEMTYPEREAL) -1;
    for(int index=0; index<a_parVars->m_total; ++index)
    {
      if(PartitionVars::NOT_TAKEN == a_parVars->m_partition[index])
      {
        Rect* curRect = &a_parVars->m_branchBuf[index].m_rect;
        Rect rect0 = CombineRect(curRect, &a_parVars->m_cover[0]);
        Rect rect1 = CombineRect(curRect, &a_parVars->m_cover[1]);
        ELEMTYPEREAL growth0 = CalcRectVolume(&rect0) - a_parVars->m_area[0];
        ELEMTYPEREAL growth1 = CalcRectVolume(&rect1) - a_parVars->m_area[1];
        ELEMTYPEREAL diff = growth1 - growth0;
        if(diff >= 0)
        {
          group = 0;
        }
        else
        {
          group = 1;
          diff = -diff;
        }

        if(diff > biggestDiff)
        {
          biggestDiff = diff;
          chosen = index;
          betterGroup = group;
        }
        else if((diff == biggestDiff) && (a_parVars->m_count[group] < a_parVars->m_count[betterGroup]))
        {
          chosen = index;
          betterGroup = group;
        }
      }
    }
    Classify(chosen, betterGroup, a_parVars);
  }

  // If one group too full, put remaining rects in the other
  if((a_parVars->m_count[0] + a_parVars->m_count[1]) < a_parVars->m_total)
  {
    if(a_parVars->m_count[0] >= a_parVars->m_total - a_parVars->m_minFill)
    {
      group = 1;
    }
    else
    {
      group = 0;
    }
    for(int index=0; index<a_parVars->m_total; ++index)
    {
      if(PartitionVars::NOT_TAKEN == a_parVars->m_partition[index])
      {
        Classify(index, group, a_parVars);
      }
    }
  }

  ASSERT((a_parVars->m_count[0] + a_parVars->m_count[1]) == a_parVars->m_total);
  ASSERT((a_parVars->m_count[0] >= a_parVars->m_minFill) &&
        (a_parVars->m_count[1] >= a_parVars->m_minFill));
}

RTREE_TEMPLATE
void RTREE_QUAL::LoadNodes(Node* a_nodeA, Node* a_nodeB, PartitionVars* a_parVars)
{
  ASSERT(a_nodeA);
  ASSERT(a_nodeB);
  ASSERT(a_parVars);

  for(int index=0; index < a_parVars->m_total; ++index)
  {
    ASSERT(a_parVars->m_partition[index] == 0 || a_parVars->m_partition[index] == 1);

    int targetNodeIndex = a_parVars->m_partition[index];
    Node* targetNodes[] = {a_nodeA, a_nodeB};

    bool nodeWasSplit = AddBranch(&a_parVars->m_branchBuf[index], targetNodes[targetNodeIndex], NULL);
    ASSERT(!nodeWasSplit);
  }
}

RTREE_TEMPLATE
void RTREE_QUAL::InitParVars(PartitionVars* a_parVars, int a_maxRects, int a_minFill)
{
  ASSERT(a_parVars);

  a_parVars->m_count[0] = a_parVars->m_count[1] = 0;
  a_parVars->m_area[0] = a_parVars->m_area[1] = (ELEMTYPEREAL)0;
  a_parVars->m_total = a_maxRects;
  a_parVars->m_minFill = a_minFill;
  for(int index=0; index < a_maxRects; ++index)
  {
    a_parVars->m_partition[index] = PartitionVars::NOT_TAKEN;
  }
}


RTREE_TEMPLATE
void RTREE_QUAL::PickSeeds(PartitionVars* a_parVars)
{
  int seed0 = 0, seed1 = 0;
  ELEMTYPEREAL worst, waste;
  ELEMTYPEREAL area[MAXNODES+1];

  for(int index=0; index<a_parVars->m_total; ++index)
  {
    area[index] = CalcRectVolume(&a_parVars->m_branchBuf[index].m_rect);
  }

  worst = -a_parVars->m_coverSplitArea - 1;
  for(int indexA=0; indexA < a_parVars->m_total-1; ++indexA)
  {
    for(int indexB = indexA+1; indexB < a_parVars->m_total; ++indexB)
    {
      Rect oneRect = CombineRect(&a_parVars->m_branchBuf[indexA].m_rect, &a_parVars->m_branchBuf[indexB].m_rect);
      waste = CalcRectVolume(&oneRect) - area[indexA] - area[indexB];
      if(waste > worst)
      {
        worst = waste;
        seed0 = indexA;
        seed1 = indexB;
      }
    }
  }

  Classify(seed0, 0, a_parVars);
  Classify(seed1, 1, a_parVars);
}

RTREE_TEMPLATE
void RTREE_QUAL::Classify(int a_index, int a_group, PartitionVars* a_parVars)
{
  ASSERT(a_parVars);
  ASSERT(PartitionVars::NOT_TAKEN == a_parVars->m_partition[a_index]);

  a_parVars->m_partition[a_index] = a_group;

  if (a_parVars->m_count[a_group] == 0)
  {
    a_parVars->m_cover[a_group] = a_parVars->m_branchBuf[a_index].m_rect;
  }
  else
  {
    a_parVars->m_cover[a_group] = CombineRect(&a_parVars->m_branchBuf[a_index].m_rect, &a_parVars->m_cover[a_group]);
  }

  a_parVars->m_area[a_group] = CalcRectVolume(&a_parVars->m_cover[a_group]);

  ++a_parVars->m_count[a_group];
}

RTREE_TEMPLATE
bool RTREE_QUAL::RemoveRect(Rect* a_rect, const DATATYPE& a_id, Node** a_root)
{
  ASSERT(a_rect && a_root);
  ASSERT(*a_root);

  ListNode* reInsertList = NULL;

  if(!RemoveRectRec(a_rect, a_id, *a_root, &reInsertList))
  {
    while(reInsertList)
    {
      Node* tempNode = reInsertList->m_node;

      for(int index = 0; index < tempNode->m_count; ++index)
      {
        // TODO go over this code. should I use (tempNode->m_level - 1)?
        InsertRect(tempNode->m_branch[index],
                   a_root,
                   tempNode->m_level);
      }

      ListNode* remLNode = reInsertList;
      reInsertList = reInsertList->m_next;

      FreeNode(remLNode->m_node);
      FreeListNode(remLNode);
    }
    if((*a_root)->m_count == 1 && (*a_root)->IsInternalNode())
    {
      Node* tempNode = (*a_root)->m_branch[0].m_child;

      ASSERT(tempNode);
      FreeNode(*a_root);
      *a_root = tempNode;
    }
    return false;
  }
  else
  {
    return true;
  }
}

RTREE_TEMPLATE
bool RTREE_QUAL::RemoveRectRec(Rect* a_rect, const DATATYPE& a_id, Node* a_node, ListNode** a_listNode)
{
  ASSERT(a_rect && a_node && a_listNode);
  ASSERT(a_node->m_level >= 0);

  if(a_node->IsInternalNode())  // not a leaf node
  {
    for(int index = 0; index < a_node->m_count; ++index)
    {
      if(Overlap(a_rect, &(a_node->m_branch[index].m_rect)))
      {
        if(!RemoveRectRec(a_rect, a_id, a_node->m_branch[index].m_child, a_listNode))
        {
          if(a_node->m_branch[index].m_child->m_count >= MINNODES)
          {
            a_node->m_branch[index].m_rect = NodeCover(a_node->m_branch[index].m_child);
          }
          else
          {
            ReInsert(a_node->m_branch[index].m_child, a_listNode);
            DisconnectBranch(a_node, index); // Must return after this call as count has changed
          }
          return false;
        }
      }
    }
    return true;
  }
  else // A leaf node
  {
    for(int index = 0; index < a_node->m_count; ++index)
    {
      if(a_node->m_branch[index].m_data == a_id)
      {
        DisconnectBranch(a_node, index); // Must return after this call as count has changed
        return false;
      }
    }
    return true;
  }
}


RTREE_TEMPLATE
bool RTREE_QUAL::Overlap(Rect* a_rectA, Rect* a_rectB) const
{
  ASSERT(a_rectA && a_rectB);

  for(int index=0; index < NUMDIMS; ++index)
  {
    if (a_rectA->m_min[index] > a_rectB->m_max[index] ||
        a_rectB->m_min[index] > a_rectA->m_max[index])
    {
      return false;
    }
  }
  return true;
}

RTREE_TEMPLATE
bool RTREE_QUAL::Cover_1(Rect* a_rectA, Rect* a_rectB) const
{
  ASSERT(a_rectA && a_rectB);
  bool aux=true;

  for(int index=0; index < NUMDIMS; ++index)
  {
    if (a_rectA->m_min[index] > a_rectB->m_min[index] ||
        a_rectA->m_max[index] < a_rectB->m_max[index])
    {
      aux=false;
    }
  }
  if(aux){
    //cout<<"Elemento encontrado: [";
    /*for(int index=0; index < NUMDIMS; ++index){
      cout<<a_rectB->m_min[index]<<" "<<a_rectB->m_max[index]<<" ";

    }*/
    //std::cout<<"]"<<endl<<"---------------------------------------------"<<endl;
  }
  return aux;
}

RTREE_TEMPLATE
bool RTREE_QUAL::Cover(Rect* a_rectA, Branch* a_branch) const
{
  Rect* a_rectB=&a_branch->m_rect;
  ASSERT(a_rectA && a_rectB);
  bool aux=true;

  for(int index=0; index < NUMDIMS; ++index)
  {
    if (a_rectA->m_min[index] > a_rectB->m_min[index] ||
        a_rectA->m_max[index] < a_rectB->m_max[index])
    {
      aux=false;
    }
  }
  if(aux){
    //cout<<"Elemento encontrado: [";
    //search_export=
    /*for(int index=0; index < NUMDIMS; ++index){
      cout<<a_rectB->m_min[index]<<" "<<a_rectB->m_max[index]<<" ";
    }*/
    //std::cout<<"]"<<endl<<"---------------------------------------------"<<endl;
    search_export.push_back(a_branch->m_data);
    //cout<<"ID: "<<a_branch->m_data<<endl;
  }
  return aux;
}

RTREE_TEMPLATE
void RTREE_QUAL::ReInsert(Node* a_node, ListNode** a_listNode)
{
  ListNode* newListNode;

  newListNode = AllocListNode();
  newListNode->m_node = a_node;
  newListNode->m_next = *a_listNode;
  *a_listNode = newListNode;
}


RTREE_TEMPLATE
bool RTREE_QUAL::Search_1(Node* a_node, Rect* a_rect, int& a_foundCount, std::function<bool (const DATATYPE&)> callback) const
{
  ASSERT(a_node);
  ASSERT(a_node->m_level >= 0);
  ASSERT(a_rect);

  if(a_node->IsInternalNode())
  {
    for(int index=0; index < a_node->m_count; ++index)
    {
      if(Overlap(a_rect, &a_node->m_branch[index].m_rect))
      {
        if(!Search(a_node->m_branch[index].m_child, a_rect, a_foundCount, callback))
        {
          return false;
        }
      }
    }
  }
  else
  {
    for(int index=0; index < a_node->m_count; ++index)
    {
      if(Cover(a_rect, &a_node->m_branch[index].m_rect))
      {
        DATATYPE& id = a_node->m_branch[index].m_data;
        ++a_foundCount;

          if(callback && !callback(id))
          {
            return false; // Don't continue searching
          }
      }
    }
  }
  return true; // Continue searching
}

RTREE_TEMPLATE
bool RTREE_QUAL::Search(Node* a_node, Rect* a_rect, int& a_foundCount, std::function<bool (const DATATYPE&)> callback) const
{
  ASSERT(a_node);
  ASSERT(a_node->m_level >= 0);
  ASSERT(a_rect);

  if(a_node->IsInternalNode())
  {
    for(int index=0; index < a_node->m_count; ++index)
    {
      if(Overlap(a_rect, &a_node->m_branch[index].m_rect))
      {
        if(!Search(a_node->m_branch[index].m_child, a_rect, a_foundCount, callback))
        {
          return false;
        }
      }
    }
  }
  else
  {
    // This is a leaf node
    for(int index=0; index < a_node->m_count; ++index)
    {
      if(Cover(a_rect, &a_node->m_branch[index]))
      {
        int& id = a_node->m_branch[index].m_data;
        ++a_foundCount;
        if(callback && !callback(id))
        {
          return false; // Don't continue searching
        }
      }
    }
  }

  return true; // Continue searching
}

RTREE_TEMPLATE
void RTREE_QUAL::Search_knn(const ELEMTYPE a_point[NUMDIMS], int a_k){
  search_knn_export.clear();
  NearRect* nearest_rects = new NearRect[a_k];
  Search_nn(a_point, m_root, nearest_rects, a_k);
  for(int i=0; i<a_k; ++i){
    search_knn_export.push_back(nearest_rects[i].m_data);
  }
  return;
}

RTREE_TEMPLATE
void RTREE_QUAL::Search_nn(const ELEMTYPE* a_point, Node* a_node, NearRect* a_nearRects, int &k)
{
  double distance;
  if(a_node->IsInternalNode())  // not a leaf node
  {
    BranchDist branchDistList[a_node->m_count];
    for(int index = 0; index < a_node->m_count; ++index)
    {
      branchDistList[index].m_index = index;
      branchDistList[index].m_branch = a_node->m_branch[index];
      branchDistList[index].m_branchDistance = ComputeDistance(a_point, &a_node->m_branch[index].m_rect);
    }
    SortBranchDistList(branchDistList, a_node->m_count);
     for(int index = 0; index < a_node->m_count; ++index)
    {
      if(branchDistList[index].m_branchDistance < a_nearRects[k-1].m_rectDistance)
      {
        Search_nn(a_point, a_node->m_branch[branchDistList[index].m_index].m_child, a_nearRects, k);
      }
    }
  }
  else // A leaf node
  {
    for(int index = 0; index < a_node->m_count; ++index)
    {
      distance = ComputeDistance(a_point, &a_node->m_branch[index].m_rect);
      if(distance < a_nearRects[k-1].m_rectDistance)
      {
        NearRect newNearRect;
        newNearRect.m_rectDistance = distance;
        newNearRect.m_rect = a_node->m_branch[index].m_rect;
        newNearRect.m_data = a_node->m_branch[index].m_data;
        AddNearNeighbor(a_nearRects, newNearRect, k);
      }
    }
  }
}
 RTREE_TEMPLATE
double RTREE_QUAL::ComputeDistance(const ELEMTYPE* a_point, const Rect* a_rect)
{
  int pointInside = 0; //All dimensions must be inside Rect
  for(int axis=0; axis<NUMDIMS; ++axis)
  {
    if(a_rect->m_min[axis] <= a_point[axis] && a_point[axis] <= a_rect->m_max[axis])
      pointInside++;
  }
  if(pointInside == NUMDIMS) //Point's inside Rect
    return 0.0;
   ELEMTYPE ri, sum = 0;
  for(int axis=0; axis<NUMDIMS; ++axis)
  {
    if(a_point[axis] < a_rect->m_min[axis])
      ri = a_rect->m_min[axis];
    else if(a_point[axis] > a_rect->m_max[axis])
      ri = a_rect->m_max[axis];
    else
      ri = a_point[axis];
     sum += (a_point[axis] - ri)*(a_point[axis] - ri);
  }
  return sum;
}
 RTREE_TEMPLATE
void RTREE_QUAL::SortBranchDistList(BranchDist a_branchDistList[], int a_length)
{
  BranchDist _branchDist;
  int j = 0;
  //InsertSort for sorting according to m_branchDistance
  for(int i=0; i<a_length; ++i)
  {
    _branchDist = a_branchDistList[i];
    j=i-1;
     while(j>=0 && a_branchDistList[j].m_branchDistance > _branchDist.m_branchDistance)
    {
      a_branchDistList[j+1] = a_branchDistList[j];
      j = j-1;
    }
    a_branchDistList[j+1] = _branchDist;
  }
}
 RTREE_TEMPLATE
void RTREE_QUAL::AddNearNeighbor(NearRect a_nearRects[], NearRect& a_nearRect, int& k)
{
  a_nearRects[k-1] = a_nearRect;
  NearRect _nearRect;
   int j = 0;
  //InsertSort for sorting according to m_branchDistance
  for(int i=0; i<k; ++i)
  {
    _nearRect = a_nearRects[i];
    j=i-1;
     while(j>=0 && a_nearRects[j].m_rectDistance > _nearRect.m_rectDistance)
    {
      a_nearRects[j+1] = a_nearRects[j];
      j = j-1;
    }
    a_nearRects[j+1] = _nearRect;
  }
}

#undef RTREE_TEMPLATE
#undef RTREE_QUAL

#endif //RTREE_H
