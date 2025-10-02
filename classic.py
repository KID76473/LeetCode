# This file is to practice classical algorithm
import heapq
from collections import defaultdict


# tree
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def pre_order(root):
    if root:
        print(root.value)
        pre_order(root.left)
        pre_order(root.right)

def in_order(root):
    if root:
        pre_order(root.left)
        print(root.value)
        pre_order(root.right)

def post_order(root):
    if root:
        pre_order(root.left)
        pre_order(root.right)
        print(root.value)

def level_order(root):
    q = [root]
    while q:
        cur = q.pop(0)
        if cur.left:
            q.append(cur.left)
        if cur.right:
            q.append(cur.right)
        print(cur.value)

def lowest_common_ancestor(root, t1, t2):
    if not root:
        return None
    if root.value == t1 or root.value == t2:
        return root.value
    l = lowest_common_ancestor(root.left, t1,t2)
    r = lowest_common_ancestor(root.right, t1,t2)
    if l is not None and r is not None:
        return root.value
    return l if l else r


# #       0
# #      / \
# #     1   2
# #   /  \  /
# #  2   3  3
# #  /
# # 3
# def construct_tree(r, cur, end):
#         if cur < end:
#             r.left = construct_tree(TreeNode(cur + 1), cur + 1, end)
#             r.right = construct_tree(TreeNode(cur + 2), cur + 2, end)
#             return r
#         return None
#
#
# def construct_complete_tree(values, index=0):
#     # If the current index is beyond the last index, return None.
#     if index >= len(values):
#         return None
#     # Create the node with the current value.
#     node = TreeNode(values[index])
#     # Recursively build the left and right subtrees.
#     node.left = construct_complete_tree(values, 2 * index + 1)
#     node.right = construct_complete_tree(values, 2 * index + 2)
#     return node
#
#
# # idx, length = 0, 4
# # root = construct_tree(6)
# # print("pre order")
# # pre_order(root)
# # print("in order")
# # in_order(root)
# # print("post order")
# # post_order(root)
#
# root = construct_complete_tree(range(7))
# print(f"pre order: ")
# pre_order(root)
# print("-----------------------------------")
# print(f"level order: ")
# level_order(root)
# print("-----------------------------------")
# # print(lowest_common_ancestor(root, 3, 4))

# graph

# def dijk(g, s, t):  # g is a dict {node: [adjacent node]}
#     res = {}
#     visited = {s}
#     for k in g.keys():
#         res[k] = 'inf'
#     res[s] = 0
#     length = len(g)
#     while len(visited) != length:
#         cur = min(res, key = lambda x: g[x])
#         for node in g[cur]:
#             res[node] = min(res[node], res[cur] + g[cur][node])
#             visited.add(node)
#     return res[t]

def dijkstra(edges, start):
    def find_min(graph):
        res = -1
        val = 100000000
        for k in graph.keys():
            if k not in visited and val > graph[k]:
                val = graph[k]
                res = k
        return res

    visited = set()
    g = {}
    for e in edges:
        if e[1] not in g:
            g[e[1]] = []
        if e[0] in g:
            g[e[0]].append([e[1], e[2]])
        else:
            g[e[0]] = [[e[1], e[2]]]
    print(f'g: {g}')
    num_nodes = len(g)
    min_dist = {i + 1: 10000000 for i in range(num_nodes)}
    min_dist[start] = 0
    print(f'min_dist: {min_dist}')
    for _ in range(num_nodes):
        cur = find_min(min_dist)
        print(f'cur: {cur}')
        for neighbour in g[cur]:
            node, value = neighbour[0], neighbour[1]
            if node not in visited and min_dist[cur] + value < min_dist[node]:
                min_dist[node] = min_dist[cur] + value
        visited.add(cur)
    return min_dist


def dijkstra_min_heap(edges, start):
    g = {}
    for u, v, w in edges:
        if u not in g:
            g[u] = []
        if v not in g:
            g[v] = []
        g[u].append([v, w])
    length = len(g.keys())
    min_dist = {i + 1: float('inf') for i in range(length)}
    heap = []
    heapq.heappush(heap, (0, start))
    while heap:
        cur_dist, node = heapq.heappop(heap)
        if cur_dist > min_dist[node]:
            continue
        for u, v in g[node]:
            if cur_dist + v < min_dist[u]:
                min_dist[u] = cur_dist + v
                heapq.heappush(heap, (cur_dist + v, u))
    return min_dist


# def dijkstra_min_heap(graph, start):
#     g = {}
#     for u, v, w in graph:
#         if u not in g:
#             g[u] = []
#         if v not in g:
#             g[v] = []
#         g[u].append(v, w)
#     min_dist = {i + 1: float('inf') for i in range(len(g.keys()))}
#     visited = [[0, start]]
#     heapq.heapify(visited)
#     while visited:
#         cur_val, cur_node = heapq.heappop(visited)
#         if cur_val > min_dist[cur_node]:
#             continue
#         for neighbour in g[cur_node]:
#             if cur_val + neighbour[1] < min_dist[neighbour[0]]:
#                 min_dist[neighbour[0]] = cur_val + neighbour[1]
#                 heapq.heappush(visited, [neighbour[1], neighbour[0]])
#     return min_dist

def dijkstra(edges, start):
    def find_min():
        res, val = -1, float('inf')
        for node in dist:
            if node not in visited and dist[node] < val:
                res = node
                val = dist[node]
        return res
    g = defaultdict(list)
    for u, v, w in edges:
        g[u].append((v, w))
        g[v].append((u, w))
    dist = {u: float('inf')  for u in g}
    dist[start] = 0
    visited = set()
    for _ in range(len(g)):
        cur = find_min()
        for u, v in g[cur]:
            if dist[cur] + v < dist[u]:
                dist[u] = dist[cur] + v
        visited.add(cur)
    return dist

def dijkstra_min_heap(edges, start):
    g = defaultdict(list)
    for node1, node2, weight in edges:
        g[node1].append((node2, weight))
        g[node2].append((node1, weight))
    heap = [(0, start)]
    heapq.heapify(heap)
    dist = {node: float('inf') for node in g}
    while heap:
        cur_val, cur_node = heapq.heappop(heap)
        if cur_val > dist[cur_node]:
            print('---')
            continue
        for neighbour, weight in g[cur_node]:
            if dist[cur_node] + weight < dist[neighbour]:
                dist[neighbour] = dist[cur_node] + weight
                heapq.heappush(heap, (dist[cur_node] + weight, dist[neighbour]))
    return dist


# edges = [[2,1,1],[2,3,1],[3,4,1]]
# start = 1
# print(dijkstra_min_heap(edges, start))

# topological sort
def topological_sort(g):
    res = []
    # calculate in-degree for every node
    in_degree = {k: 0 for k in g.keys()}
    for k in g.keys():
        for u in g[k]:
            in_degree[u] += 1
    # find node to start
    s = set()
    for k in in_degree.keys():
        if in_degree[k] == 0:
            s.add(k)
    # loop
    while s:
        cur = s.pop()
        res.append(cur)
        for node in g[cur]:
            in_degree[node] -= 1
            if in_degree[node] == 0:
                s.add(node)
    return res


# Unbounded knapsack problem
def knapsack_unbd(max_weight, weight, profit):
    dp = [[0 for _ in range(len(weight) + 1)] for _ in range(max_weight + 1)]
    for i in range(1, len(dp)):
        for j in range(1, len(dp[0])):
            item_index = j - 1
            if i - weight[j - 1] >= 0:
                dp[i][j] = max(dp[i - weight[item_index]][j] + profit[item_index], dp[i][j - 1])
            else:
                dp[i][j] = dp[i][j - 1]
    print(dp)
    return dp[-1][-1]


# 0/1 knapsack problem
def knapsack01(max_weight, weight, profit):
    dp = [[0 for _ in range(len(weight) + 1)] for _ in range(max_weight + 1)]
    for i in range(1, len(dp)):
        for j in range(1, len(dp[0])):
            item_index = j - 1
            if i - weight[j - 1] >= 0:
                dp[i][j] = max(dp[i - weight[item_index]][j - 1] + profit[item_index], dp[i][j - 1])
            else:
                dp[i][j] = dp[i][j - 1]
    print(dp)
    return dp[-1][-1]


# max_weight = 5
# weight = [2, 1, 3, 2, 1]
# profit = [2, 2, 2, 1, 1]
# knapsack01(max_weight, weight, profit)


def dijkstra(edges, start):
    g = defaultdict(list)
    for (u, v, w) in edges:
        g[u].append([v, w])
        g[v].append([u, w])
    # print(g)
    h = [[0, start]]
    heapq.heapify(h)
    dist = {u: float('inf') for u in g}
    dist[start] = 0
    while h:
        cur_dist, cur = heapq.heappop(h)
        if cur_dist > dist[cur]:
            continue
        print(f'node: {cur}, dist: {cur_dist}')
        for nei, nei_dist in g[cur]:
            new_dist = cur_dist + nei_dist
            if new_dist < dist[nei]:
                dist[nei] = new_dist
                heapq.heappush(h, [new_dist, nei])
    return dist


edges = [[1,3,1],[1,4,2],[1,2,5],[1,3,1],[2,3,2],[2,4,3],[3,5,7],[4,5,2]]
start = 1
print(dijkstra(edges, start))