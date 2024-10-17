#include <iostream>
#include <vector>
#include <climits>

using namespace std;

// A structure to represent an edge in the graph
struct Edge {
    int u, v, weight;
};

// Function to implement Bellman-Ford algorithm
void bellmanFord(int source, vector<Edge>& edges, int V, int E) {
    // Create a distance vector initialized to infinity
    vector<int> dist(V, INT_MAX);

    // Set the distance to the source to 0
    dist[source] = 0;

    // Relax all edges (V - 1) times
    for (int i = 1; i <= V - 1; i++) {
        for (int j = 0; j < E; j++) {
            int u = edges[j].u;
            int v = edges[j].v;
            int weight = edges[j].weight;
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
            }
        }
    }

    // Check for negative-weight cycles
    for (int j = 0; j < E; j++) {
        int u = edges[j].u;
        int v = edges[j].v;
        int weight = edges[j].weight;
        if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
            cout << "Graph contains a negative-weight cycle.\n";
            return;
        }
    }

    // Print the calculated shortest distances
    cout << "Vertex\tDistance from Source\n";
    for (int i = 0; i < V; i++) {
        if (dist[i] == INT_MAX)
            cout << i << "\t\t" << "INF" << endl;
        else
            cout << i << "\t\t" << dist[i] << endl;
    }
}

int main() {
    int V, E, u, v, w;

    // Input number of vertices and edges
    cout << "Enter number of vertices and edges: ";
    cin >> V >> E;

    // Create a list to store all edges
    vector<Edge> edges(E);

    // Input all edges
    cout << "Enter edges (u v w) where u and v are vertices and w is the weight:\n";
    for (int i = 0; i < E; i++) {
        cin >> u >> v >> w;
        edges[i] = {u, v, w};
    }

    int source;
    cout << "Enter the source vertex: ";
    cin >> source;

    // Call Bellman-Ford algorithm
    bellmanFord(source, edges, V, E);

    return 0;
}
