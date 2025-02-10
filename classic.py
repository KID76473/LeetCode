# This file is to practice classical algorithm


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
def dijk(g, s, t):  # g is a dict {node: [adjacent node]}
    res = {}
    visited = {s}
    for k in g.keys():
        res[k] = 'inf'
    res[s] = 0
    length = len(g)
    while len(visited) != length:
        cur = min(res, key = lambda x: g[x])
        for node in g[cur]:
            res[node] = min(res[node], res[cur] + g[cur][node])
            visited.add(node)
    return res[t]
