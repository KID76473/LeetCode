import collections
from typing import Optional, List
from leetcode_sol import Solutions_LeetCode
from online_assessment import Solution_oneline_assessment
from huawei import huawei_solution


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def main():
    # linked list
    def build_linked_list(values):
        if not values:
            return None
        head = ListNode(values[0])
        current = head
        for value in values[1:]:
            current.next = ListNode(value)
            current = current.next
        return head

    def print_linked_list(head):
        current = head
        while current:
            print(current.val, end=" -> ")
            current = current.next
        print("None")

    # # "3[a]2[bc]" aaa bcbc
    # # "3[a2[c]]" acc acc acc
    # # "2[abc]3[cd]ef" abcabc cdcdcd ef
    # values = [1,2,3,4,5]
    # a = build_linked_list(values)
    # # b = "bca"
    # a = Solutions().pairSum(a)
    # print_linked_list(a)

    # binary tree
    # build a binary tree based on the list given
    def build_tree(arr: List[int]) -> TreeNode():
        if not arr:
            return None
        root = TreeNode(arr.pop(0))
        q = [root]
        while arr:
            cur = q.pop(0)
            if arr:
                node = TreeNode(arr.pop(0))
                cur.left = node
                q.append(node)
            if arr:
                node = TreeNode(arr.pop(0))
                cur.right = node
                q.append(node)
        return root

    def dfs(root: TreeNode()):
        if root:
            print(root.val)
            dfs(root.left)
            dfs(root.right)

    nums = [1,1,1,1,1]
    nums = [0]
    target = 5
    target = 0
    print(Solutions_LeetCode().findTargetSumWays(nums, target))


if __name__ == "__main__":
    main()
