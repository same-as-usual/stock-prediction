import heapq

def best_first_search(graph, start, goal):
    priority_queue = []  # Min-heap priority queue
    heapq.heappush(priority_queue, (0, start))  # (heuristic, node)
    visited = set()
    parent = {start: None}
    
    while priority_queue:
        _, current = heapq.heappop(priority_queue)
        
        if current in visited:
            continue
        
        visited.add(current)
        print(current, end=' ')  # Print path traversal
        
        if current == goal:
            print("\nGoal reached!")
            return reconstruct_path(parent, start, goal)
        
        for neighbor, cost in graph.get(current, []):
            if neighbor not in visited:
                heapq.heappush(priority_queue, (cost, neighbor))
                parent[neighbor] = current
    
    print("\nGoal not reachable.")
    return None  # No path found

def reconstruct_path(parent, start, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    return path[::-1]  # Reverse the path

# Example graph (Adjacency list with heuristic values as cost)
graph = {
    'A': [('B', 2), ('C', 1)],
    'B': [('D', 3), ('E', 1)],
    'C': [('F', 4)],
    'D': [('G', 5)],
    'E': [('G', 2)],
    'F': [('G', 3)],
    'G': []
}

# Running Best First Search from 'A' to 'G'
print("Best First Search Path:")
path = best_first_search(graph, 'A', 'G')
print("Path:", path)
