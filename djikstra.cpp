#include <iostream>
#include <vector>
#include <queue>
#include <climits>

using namespace std;

// Define a pair for the priority queue (distance, node)
typedef pair<int, int> pii;

// Function to implement Dijkstra's algorithm
void dijkstra(int source, vector<vector<pii>>& graph, int V) {
    // Create a distance vector initialized to infinity
    vector<int> dist(V, INT_MAX);
    
    // Priority queue to store the vertices that are being processed
    priority_queue<pii, vector<pii>, greater<pii>> pq;
    
    // Start with the source node
    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        // Get the vertex with the smallest distance
        int u = pq.top().second;
        pq.pop();

        // Iterate through all the adjacent vertices of u
        for (auto& edge : graph[u]) {
            int v = edge.second;
            int weight = edge.first;

            // If there is a shorter path to v through u
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }

    // Print the calculated shortest distances
    cout << "Vertex\tDistance from Source\n";
    for (int i = 0; i < V; i++) {
        cout << i << "\t\t" << dist[i] << endl;
    }
}

int main() {
    int V, E, u, v, w;
    
    // Input number of vertices and edges
    cout << "Enter number of vertices and edges: ";
    cin >> V >> E;

    // Create a graph as an adjacency list
    vector<vector<pii>> graph(V);

    // Input all edges
    cout << "Enter edges (u v w) where u and v are vertices and w is the weight:\n";
    for (int i = 0; i < E; i++) {
        cin >> u >> v >> w;
        graph[u].push_back({w, v});
        graph[v].push_back({w, u}); // If the graph is undirected
    }

    int source;
    cout << "Enter the source vertex: ";
    cin >> source;

    // Call Dijkstra's algorithm
    dijkstra(source, graph, V);

    return 0;
}
