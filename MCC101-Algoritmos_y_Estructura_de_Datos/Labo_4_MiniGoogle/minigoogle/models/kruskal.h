// C++ program for Kruskal's algorithm to find Minimum
// Spanning Tree of a given connected, undirected and
// weighted graph
#include "Includes.h"

// Creating shortcut for an integer pair
typedef  pair<int, int> iPair;
typedef pair<int, iPair> wPair;

// To represent Disjoint Sets
class DisjointSets
{
    private:
        int *parent, *rnk;
        int n;
    public:
        // Constructor.
        DisjointSets(int n)
        {
            // Allocate memory
            this->n = n;
            parent = new int[n+1];
            rnk = new int[n+1];
    
            // Initially, all vertices are in
            // different sets and have rank 0.
            for (int i = 0; i <= n; i++)
            {
                rnk[i] = 0;
    
                //every element is parent of itself
                parent[i] = i;
            }
        }
    
        // Find the parent of a node 'u'
        // Path Compression
        int find(int u)
        {
            /* Make the parent of the nodes in the path
            from u--> parent[u] point to parent[u] */
            if (u != parent[u])
                parent[u] = find(parent[u]);
            return parent[u];
        }
    
        // Union by rank
        void merge(int x, int y)
        {
            x = find(x), y = find(y);
    
            /* Make tree with smaller height
            a subtree of the other tree  */
            if (rnk[x] > rnk[y])
                parent[y] = x;
            else // If rnk[x] <= rnk[y]
                parent[x] = y;
    
            if (rnk[x] == rnk[y])
                rnk[y]++;
        }
};

// Structure to represent a graph
class Graph{
    private:
        int V, E;
        vector< pair<int, iPair> > edges;
    
    public:
        // Constructor
        Graph(int V, int E)
        {
            this->V = V;
            this->E = E;
        }
    
        // Utility function to add an edge
        void addEdge(int u, int v, int w)
        {
            edges.push_back({w, {u, v}});
        }
    
        // Function to find MST using Kruskal's
        // MST algorithm
        vector< pair<int, iPair> > kruskalMST();
};

 /* Functions returns weight of the MST*/
 
vector< pair<int, iPair> > Graph::kruskalMST()
{
    vector< pair<int, iPair> > MST; 
    //int mst_wt = 0; // Initialize result
    //int max = 0;
    //vector< pair<int, iPair> >::iterator maxEdge;
    // Sort edges in increasing order on basis of cost
    sort(edges.begin(), edges.end());
 
    // Create disjoint sets
    DisjointSets ds(V);
 
    // Iterate through all sorted edges
    vector< pair<int, iPair> >::iterator it;
    for (it=edges.begin(); it!=edges.end(); it++)
    {
        int u = it->second.first;
        int v = it->second.second;
 
        int set_u = ds.find(u);
        int set_v = ds.find(v);
 
        // Check if the selected edge is creating
        // a cycle or not (Cycle is created if u
        // and v belong to same set)
        if (set_u != set_v)
        {
            // Current edge will be in the MST
            // so print it
            MST.push_back(wPair(it->first,iPair(u,v)));
            cout << u << " - " << v << endl;
 
            // Update MST weight
            //mst_wt += it->first;
            /*if(max < it->first){
                max = it->first;
                maxEdge = it;
            }*/
            // Merge two sets
            ds.merge(set_u, set_v);
        }
    }
 
    return MST;
}


 
