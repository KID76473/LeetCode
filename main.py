import collections
from typing import Optional

from solutions import Solutions


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

    def build_tree(arr: int[]) -> TreeNode():
        root = TreeNode()
        cur = root
        for x in arr:
            while cur:
                if not cur.left:
                    cur.left = TreeNode(x)
                if not cur.right:
                    cur.right = TreeNode(x)

    a = [3,9,20,None,None,15,7]
    print(Solutions().maxDepth(a))


if __name__ == "__main__":
    main()
