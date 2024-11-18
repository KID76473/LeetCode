import collections
import math
import heapq
from typing import List
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solutions:
    # 121
    def maxProfit(self, prices: List[int]) -> int:
        min_price = 10000
        max_profit = 0
        for p in prices:
            min_price = min(p, min_price)
            max_profit = max(p - min_price, max_profit)
        return max_profit

    # 238
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        product = 1
        result = []
        for n in nums:
            result.append(int(product))
            product *= n
        product = 1
        for i in range(len(nums) - 1, -1, -1):
            result[i] *= product
            product *= nums[i]
        return result
    # zeros = 0
    # for n in nums:
    #     if n == 0:
    #         zeros += 1
    # if zeros == 0:
    #     result = []
    #     product = 1
    #     for n in nums:
    #         product *= n
    #     for n in nums:
    #         result.append(int(product / n))
    #     return result
    # elif zeros == 1:
    #     result = []
    #     product = 1
    #     for n in nums:
    #         if n != 0:
    #             product *= n
    #     for n in nums:
    #         result.append(0) if n != 0 else result.append(product)
    #     return result
    # else:
    #     return [0] * len(nums)

    # 53
    def maxSubArray(self, nums: List[int]) -> int:
        # max_sum = cur_sum = nums[0]
        # for n in nums:
        #     cur_sum = max(n, cur_sum + n)
        #     max_sum = max(max_sum, cur_sum)
        # return max_sum

        for i in range(1, len(nums)):
            if nums[i - 1] > 0:
                nums[i] += nums[i - 1]
        return max(nums)

    # 191 NOT FINISHED
    def hammingWeight(self, n: int) -> int:
        num = 0
        while n >= 1:
            if int(n % 10) == 1:
                num += 1
            n = int(n / 10)
        return num

    # 70
    def climbStairs(self, n: int) -> int:
        # if n == 0 or n == 1:
        #     return 1
        # elif n == 2:
        #     return 2
        # else:
        #     return self.climbStairs(n - 1) + self.climbStairs(n - 2)

        dp = [1, 2]
        for i in range(2, n):
            dp.append(dp[i - 1] + dp[i - 2])
        return dp[n - 1]

    # 322
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1

    # 300
    # DOT NOT UNDERSTAND DICHOTOMY
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1]
        for i in range(len(nums)):
            dp.append(1)
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[j] + 1, dp[i])
        return max(dp)
        # length = max_length = 1
        # prev = nums[0]
        # for i in range(0, len(nums) - 1):
        #     if nums[i + 1] > nums[i]:
        #         length += 1
        #         prev = nums[i]
        #     elif nums[i + 1] > prev:
        #         length += 1
        #         length = max_length
        #     else:
        #         length = 1
        #     max_length = max(length, max_length)
        #     print(1, ": ", max_length)
        # return max_length

    # 1143
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = []
        # generate 2d array
        # for i in range(m + 1):
        #     dp.append([])
        #     for j in range(n + 1):
        #         dp[i].append(0)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # fill in 2d array
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        # print out dp
        for i in range(len(dp)):
            print(dp[i])
        return dp[m][n]

    # 1768. Merge Strings Alternately
    def mergeAlternately(self, word1: str, word2: str) -> str:
        result = ''
        s1, s2 = word1, word2
        count = -1
        while len(s1) > 0 and len(s2) > 0:
            if count == -1:
                result += s1[0: 1]
                s1 = s1[1:]
            else:
                result += s2[0: 1]
                s2 = s2[1:]
            count *= -1
        if len(s1) > 0:
            result += s1
        elif len(s2) > 0:
            result += s2
        return result

    # 1071. Greatest Common Divisor of Strings
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        if str1 + str2 != str2 + str1:
            return ""
        return str1[0: math.gcd(len(str1), len(str2))]

    # 1431. Kids With the Greatest Number of Candies
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        result = []
        king = -1
        for k in candies:
            if k > king:
                king = k
        for i in range(len(candies)):
            if candies[i] + extraCandies >= king:
                result.append(True)
            else:
                result.append(False)
        return result

    # 345. Reverse Vowels of a String
    def reverseVowels(self, s: str) -> str:
        vowels = ""
        consonants = ""
        result = ""
        for c in s:
            if c in ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']:
                vowels += c
                consonants += '*'
            else:
                consonants += c
        i = len(vowels) - 1
        for c in consonants:
            if c == '*':
                result += vowels[i]
                i -= 1
            else:
                result += c
        return result

    # 605. Can Place Flowers
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        plantable = 0
        length = len(flowerbed)
        for i in range(length):
            if flowerbed[i] == 0 and (i == 0 or flowerbed[i - 1] == 0) and (i == length - 1 or flowerbed[i + 1] == 0):
                plantable += 1
                flowerbed[i] = 1
        return plantable >= n
        # zeros = 0
        # length = 0
        # zero_list = []
        # flowerbed = [1, 0] + flowerbed + [0, 1]
        # for p in flowerbed:
        #     if p == 0:
        #         zeros += 1
        #     else:
        #         if zeros != 0:
        #             zero_list.append(zeros)
        #             length += 1
        #             zeros = 0
        # # print(length, zero_list)
        # plantable = 0
        # for z in zero_list:
        #     plantable += math.floor((z - 1) / 2)
        #     print(plantable)
        # return plantable >= n

    # 345. Reverse Vowels of a String
    def reverseVowels(self, s: str) -> str:
        vowels = ""
        consonants = ""
        result = ""
        for c in s:
            if c in ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']:
                vowels += c
                consonants += '*'
            else:
                consonants += c
        i = len(vowels) - 1
        for c in consonants:
            if c == '*':
                result += vowels[i]
                i -= 1
            else:
                result += c
        return result

    # 151. Reverse Words in a String
    def reverseWords(self, s: str) -> str:
        if s[-1: 0] != " ":  # add a space after string
            s += " "
        words = []
        word = ""
        for c in s:
            if c != ' ':
                word += c
            elif word != "":  # append the word to words only if it meets the first space
                words.append(word)
                word = ""
        result = ""
        length = len(words)
        for _ in range(length):  # reload string with reversed order
            result += words.pop() + " "
        result = result[0: -1]  # eliminate the last space
        return result

    # 334. Increasing Triplet Subsequence
    def increasingTriplet(self, nums: List[int]) -> bool:
        first = second = float('inf')
        for num in nums:
            if num <= first:
                first = num
            elif num <= second:
                second = num
            else:
                return True
        return False
        # length = len(nums)
        # current_min, current_max = math.inf, -math.inf
        # left_min = [math.inf] * length
        # right_max = [-math.inf] * length
        # for i in range(length):
        #     if current_min > nums[i]:
        #         current_min = nums[i]
        #     left_min[i] = current_min
        #     if current_max < nums[-1 - i]:
        #         current_max = nums[-1 - i]
        #     right_max[-1 - i] = current_max
        # print(left_min, right_max)
        # for i in range(length):
        #     if left_min[i] < nums[i] < right_max[i]:
        #         return True
        # return False

    # UNFINISHED!!!!!!!!!!!!!
    # 443. String Compression
    def compress(self, chars: List[str]) -> int:
        chars += "!"  # place holder
        result = []
        last = chars[0]
        count = 0
        for c in chars:
            if c == last:
                count += 1
            else:
                result.append(last)
                if count > 1:
                    result.append(str(count))
                count = 1
            last = c
        chars = result
        print(chars)
        return len(result)

    # 283. Move Zeroes
    def moveZeroes(self, nums: List[int]) -> None:
        zeros = []
        for i in range(len(nums)):
            if nums[i] == 0:
                zeros.append(i)
            elif len(zeros) != 0:
                nums[zeros[0]] = nums[i]
                nums[i] = 0
                zeros.remove(zeros[0])
                zeros.append(i)
        # print(nums)

    # 392. Is Subsequence
    def isSubsequence(self, s: str, t: str) -> bool:
        i, j = 0, 0
        while i < len(s) and j < len(t):
            print(s[i], i, t[j], j)
            if s[i] == t[j]:
                i += 1
            j += 1
        return i == len(s)

    # 1679. Max Number of K-Sum Pairs
    def maxOperations(self, nums: List[int], k: int) -> int:
        # O(n*logn)
        nums.sort()
        i, j, opt = 0, len(nums) - 1, 0
        while i < j:
            if nums[i] + nums[j] > k:
                j -= 1
            elif nums[i] + nums[j] < k:
                i += 1
            else:
                nums.pop(j)
                nums.pop(i)
                opt += 1
                j -= 2
                print(nums)
        return opt
        # # O(n^2)
        # i, opt = 0, 0
        # while len(nums) > 0 and i < len(nums):
        #     if k > nums[i]:
        #         res = k - nums[i]
        #         for j in range(i + 1, len(nums)):
        #             if nums[j] == res:
        #                 nums.pop(j)
        #                 nums.pop(i)
        #                 opt += 1
        #                 i -= 1
        #                 break
        #     i += 1
        # return opt

    # 643. Maximum Average Subarray I
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        cur = mx = sum(nums[: k])
        for i in range(len(nums) - k):
            cur = cur - nums[i] + nums[i + k]
            mx = max(mx, cur)
        return mx / k

    # 1456. Maximum Number of Vowels in a Substring of Given Length
    def maxVowels(self, s: str, k: int) -> int:
        vowels = 0
        for i in range(k):
            if s[i: i + 1] in ['a', 'e', 'i', 'o', 'u']:
                vowels += 1
        mx = vowels
        print(vowels)
        for i in range(len(s) - k):
            if s[i: i + 1] in ['a', 'e', 'i', 'o', 'u']:
                vowels -= 1
            if s[i + k: i + k + 1] in ['a', 'e', 'i', 'o', 'u']:
                vowels += 1
            mx = max(mx, vowels)
            print(i, vowels)
        return mx

    # 1004. Max Consecutive Ones III
    def longestOnes(self, nums: List[int], k: int) -> int:
        l, r = 0, 0
        for r in range(len(nums)):
            if nums[r] == 0:
                k -= 1
            if k < 0:
                if nums[l] == 0:
                    k += 1
                l += 1
        return r - l + 1

        # O(n^2)
        # mx = 0
        # for i in range(len(nums)):
        #     cur = 0
        #     j = i
        #     res = k
        #     while j < len(nums) and (res > 0 or nums[j] == 1):
        #         if nums[j] == 0 and res > 0:
        #             cur += 1
        #             res -= 1
        #         elif nums[j] == 1:
        #             cur += 1
        #         j += 1
        #     mx = max(mx, cur)
        # return mx

    # 1493. Longest Subarray of 1's After Deleting One Element
    def longestSubarray(self, nums: List[int]) -> int:
        l, r, k = 0, 0, 1
        for r in range(len(nums)):
            if nums[r] == 0:
                k -= 1
            if k < 0:
                if nums[l] == 0:
                    k += 1
                l += 1
        return r - l

    # 1732. Find the Highest Altitude
    def largestAltitude(self, gain: List[int]) -> int:
        cur, mx = 0, 0
        for num in gain:
            cur += num
            mx = max(cur, mx)
        return mx

    # 724. Find Pivot Index
    def pivotIndex(self, nums: List[int]) -> int:
        left_sum, right_sum = 0, sum(nums)
        for index, num in enumerate(nums):
            right_sum -= num
            if left_sum == right_sum:
                return index
            left_sum += num
        return -1
        # cur, left, right = 0, [], []
        # for num in nums:
        #     left.append(cur)
        #     cur += num
        # cur = 0
        # for i in range(len(nums)):
        #     right.append(cur)
        #     cur += nums[-i - 1]
        # right.reverse()
        # for i in range(len(nums)):
        #     if left[i] == right[i]:
        #         return i
        # return -1

    # 2215. Find the Difference of Two Arrays
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        s1, s2 = set(nums1), set(nums2)
        return [list(s1 - s2), list(s2 - s1)]

        # result = [[], []]
        # for num1 in nums1:
        #     if num1 not in nums2 and num1 not in result[0]:
        #         result[0].append(num1)
        # for num2 in nums2:
        #     if num2 not in nums1 and num2 not in result[1]:
        #         result[1].append(num2)
        # return result

    # 1207. Unique Number of Occurrences
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        occurrences = collections.Counter(arr)
        counts = list(occurrences.values())
        unique_counts = set(counts)
        return len(counts) == len(unique_counts)

        # mp = {}
        # for a in arr:
        #     if a in mp.keys():
        #         mp[a] += 1
        #     else:
        #         mp[a] = 0
        # print(mp)
        # values = set()
        # last = 0
        # for key in mp.keys():
        #     values.add(mp[key])
        #     if len(values) == last:
        #         return False
        #     last = len(values)
        # return True

    # 1657. Determine if Two Strings Are Close
    def closeStrings(self, word1: str, word2: str) -> bool:
        if len(word1) != len(word2):
            return False
        count1 = collections.Counter(word1)
        count2 = collections.Counter(word2)
        if count1.keys() == count2.keys() and sorted(count1.values()) == sorted(count2.values()):
            return True
        return False

    # 2352. Equal Row and Column Pairs
    def equalPairs(self, grid: List[List[int]]) -> int:
        cnt = 0
        row_dict = collections.defaultdict(int)
        for row in grid:
            row_dict[str(row)] += 1
        length = len(grid)
        for i in range(length):
            col = []
            for j in range(length):
                col.append(grid[j][i])
            cnt += row_dict[str(col)]
        return cnt
        
        # length = len(grid)
        # result = 0
        # columns = []
        # for i in range(length):
        #     col = []
        #     for j in range(length):
        #         col.append(grid[j][i])
        #     columns.append(col)
        # for i in range(length):
        #     for j in range(length):
        #         if grid[i] == columns[j]:
        #             result += 1
        # return result

    # 2390. Removing Stars From a String
    def removeStars(self, s: str) -> str:
        stack = []
        for c in s:
            if c == '*':
                stack.pop()
            else:
                stack.append(c)
        return ''.join(stack)

    # 735. Asteroid Collision
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        for ast in asteroids:
            if ast < 0:
                explode = False
                while len(stack) >= 1 and stack[-1] > 0 and not explode:
                    temp = stack.pop()
                    if abs(temp) > abs(ast):
                        stack.append(temp)
                        explode = True
                    elif abs(temp) == abs(ast):
                        explode = True
                if not explode:
                    stack.append(ast)
            else:
                stack.append(ast)
        return stack

        # stack = []
        # last = asteroids[0]
        # i = 0
        # while i < len(asteroids) or len(stack) >= 2 and stack[-1] < 0 < stack[-2]:
        #     if len(stack) >= 2 and stack[-1] < 0 < stack[-2]:
        #         temp1 = stack.pop()
        #         temp2 = stack.pop()
        #         if abs(temp1) > abs(temp2):
        #             stack.append(temp1)
        #         elif abs(temp1) < abs(temp2):
        #             stack.append(temp2)
        #     elif i < len(asteroids):
        #         if asteroids[i] < 0 < last and len(stack) > 0:
        #             temp = stack.pop()
        #             if abs(asteroids[i]) > abs(temp):
        #                 stack.append(asteroids[i])
        #             elif abs(asteroids[i]) < abs(temp):
        #                 stack.append(temp)
        #         else:
        #             stack.append(asteroids[i])
        #         if len(stack) > 0:
        #             last = stack[-1]
        #         i += 1
        #     print(stack)
        # return stack
    # 394. Decode String
    def decodeString(self, s: str) -> str:
        return "NOT FINISHED"

    # 649. Dota2 Senate
    def predictPartyVictory(self, senate: str) -> str:
        # brute force
        # queue = [True] * len(senate)
        # last = ''
        # while sum(queue) > 1:
        #     for i in range(len(senate)):
        #         if queue[i]:
        #             last = senate[i]
        #             for j in range(len(senate)):
        #                 if senate[i] != senate[j] and queue[j]:
        #                     queue[j] = False
        #                     break
        # if last == '':
        #     return senate[0]
        # if last == 'R':
        #     return "Radiant"
        # elif last == 'D':
        #     return "Dire"
        length = len(senate)
        qd, qr = [], []
        for i in range(len(senate)):
            if senate[i] == 'R':
                qr.append(i)
            else:
                qd.append(i)
        while len(qd) != 0 and len(qr) != 0:
            if qd[0] < qr[0]:
                qd.append(length + qd[0])
                qd.pop(0)
                qr.pop(0)
            else:
                qr.append(length + qr[0])
                qr.pop(0)
                qd.pop(0)

        if len(qd) == 0:
            return "Radiant"
        else:
            return "Dire"

    # 2095. Delete the Middle Node of a Linked List
    # Definition for singly-linked list.

    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        if not cur.next:
            return None

        last = head
        cur = head
        cur2 = head
        while cur2.next and cur2.next.next:
            last = cur
            cur = cur.next
            cur2 = cur2.next.next
        if cur2.next:
            cur.next = cur.next.next
        else:
            last.next = last.next.next
        return head

    # 328. Odd Even Linked List
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # # to slow and uses too much space
        # if not head:
        #     return head
        # if head.next:
        #     odd, even = ListNode(0, None), ListNode(0, None)
        # else:
        #     return head
        # cur, cur_odd, cur_even = head, odd, even
        # temp = True
        # while cur:
        #     if temp:
        #         cur_odd.next = ListNode(cur.val, None)
        #         cur_odd = cur_odd.next
        #         temp = not temp
        #     else:
        #         cur_even.next = ListNode(cur.val, None)
        #         cur_even = cur_even.next
        #         temp = not temp
        #     cur = cur.next
        # odd = odd.next
        # even = even.next
        # cur = odd
        # while cur.next:
        #     cur = cur.next
        # cur.next = even
        # return odd

        if not head:
            return head
        odd, even, even_head = head, head.next, head.next
        while even and even.next:
            odd.next = odd.next.next
            odd = odd.next
            even.next = even.next.next
            even = even.next
        odd.next = even_head
        return head

    # 206. Reverse Linked List
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        cur = head
        q = []
        while cur:
            q.append(cur.val)
            cur = cur.next
        # print(q)
        res = ListNode()
        cur = res
        while len(q) > 0:
            temp = q.pop()
            # print(temp)
            cur.next = ListNode(temp, None)
            cur = cur.next
        res = res.next
        return res

    # 2130. Maximum Twin Sum of a Linked List
    def pairSum(self, head: Optional[ListNode]) -> int:
        # # use array
        # arr = []
        # cur = head
        # while cur:
        #     arr.append(cur.val)
        #     cur = cur.next
        # i, j = 0, len(arr) - 1
        # res = 0
        # while i < j:
        #     if arr[i] + arr[j] > res:
        #         res = arr[i] + arr[j]
        #     i += 1
        #     j -= 1
        # return res

        # reverse the second part
        slow, fast = head, head
        maxVal = 0

        # Get middle of linked list
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        # Reverse second part of linked list
        curr, prev = slow, None

        while curr:
            curr.next, prev, curr = prev, curr, curr.next

            # Get max sum of pairs
        while prev:
            maxVal = max(maxVal, head.val + prev.val)
            prev = prev.next
            head = head.next

        return maxVal

    # 104. Maximum Depth of Binary Tree
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # # too slow and use too much space
        # def helper(depth: int, root: Optional[TreeNode]):
        #     depth_l, depth_r = depth, depth
        #     if root:
        #         if root.left:
        #             depth_l = 1 + helper(depth, root.left)
        #         if root.right:
        #             depth_r = 1 + helper(depth, root.right)
        #     return max(depth_l, depth_r)
        # if not root:
        #     return 0
        # return helper(1, root)

        # simpler
        if not root:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left, right) + 1

    # 872. Leaf-Similar Trees
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def helper(arr: List[int], root: Optional[TreeNode]):
            if root:
                if not root.left and not root.right:
                    arr.append(root.val)
                helper(arr, root.left)
                helper(arr, root.right)
                # if root.left:
                #     helper(arr, root.left)
                # if root.right:
                #     helper(arr, root.right)
                # else:
                #     arr.append(root.val)
        arr1 = []
        helper(arr1, root1)
        arr2 = []
        helper(arr2, root2)
        return arr1 == arr2

    # 1448. Count Good Nodes in Binary Tree
    def goodNodes(self, root: TreeNode) -> int:
        def helper(max: int, root: TreeNode) -> int:
            if root.val >= max:
                max = root.val
                num[0] += 1
            if root.left:
                helper(max, root.left)
            if root.right:
                helper(max, root.right)

        num = [0]
        helper(root.val, root)
        return num[0]

    # 1372. Longest ZigZag Path in a Binary Tree
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        longest = [0]
        def helper(root: Optional[TreeNode], left: bool, length: int):
            longest[0] = max(longest[0], length)
            if left:
                if root.left:
                    helper(root.left, False, length + 1)
                if root.right:
                    helper(root.right, True, 1)
            else:
                if root.left:
                    helper(root.left, False, 1)
                if root.right:
                    helper(root.right, True, length + 1)

        helper(root, True, 0)
        helper(root, False, 0)
        return longest[0]

    # 199. Binary Tree Right Side View
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        def helper(root: Optional[TreeNode], level: int):
            if not root:
                return
            if level == len(result):
                result.append(root.val)
            helper(root.right, level + 1)
            helper(root.left, level + 1)
        helper(root, 0)
        return result

    # 236. Lowest Common Ancestor of a Binary Tre
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        if root == p or root == q:
            return root
        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)
        if l and r:
            return root
        return l if l else r

    # 1161. Maximum Level Sum of a Binary Tree
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        q = [root]
        s = root.val
        level, j = 1, 1
        while len(q) > 0:
            i, length = 0, len(q)
            cur_val = 0
            while i < length:
                cur = q.pop(0)
                cur_val += cur.val
                i += 1
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            if cur_val > s:
                s = cur_val
                level = j
            j += 1
        return level

    # 700. Search in a Binary Search Tree
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return
        elif val < root.val:
            return self.searchBST(root.left, val)
        elif val > root.val:
            return self.searchBST(root.right, val)
        return root

    # 450. Delete Node in a BST
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.left:
                return root.right
            if not root.right:
                return root.left
            if root.left and root.right:
                temp = root.right
                while temp.left:
                    temp = temp.left
                root.val = temp.val
                root.right = self.deleteNode(root.right, root.val)
        return root

    # 841. Keys and Rooms
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        s = set()
        s.add(0)
        q = [0]
        length = len(rooms)
        while q:
            index = q.pop()
            for r in rooms[index]:
                if r not in s:
                    s.add(r)
                    q.append(r)
            if len(s) == length:
                return True
        return False

    # 547. Number of Provinces
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        num = 0
        visited = [False] * len(isConnected)
        for i in range(len(isConnected)):  # loop thru every unvisited node
            if not visited[i]:
                visited[i] = True
                s = set()
                q = [i]
                while q:  # dfs current node and its adjacent nodes
                    cur = q.pop()
                    if cur not in s:
                        for j in range(len(isConnected[cur])):
                            if isConnected[cur][j] and j not in s:
                                q.append(j)
                                visited[j] = True
                        s.add(cur)
                num += 1
        return num

    # week 2
    # 1466. Reorder Routes to Make All Paths Lead to the City Zero
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        # # O(n^2) without using adj list
        # count = 0
        # s = [0]
        # visited = []
        # while s:
        #     cur = s.pop()
        #     visited.append(cur)
        #     for e in connections:
        #         if e[0] == cur and e[1] not in visited:
        #             # print(e)
        #             count += 1
        #             s.append(e[1])
        #             # visited.append(e[1])
        #         elif e[1] == cur and e[0] not in visited:
        #             s.append(e[0])
        #             # visited.append(e[0])
        # return count

        # O(n) using adj list
        changes = [0]
        adjacent_list = [[] for _ in range(n)]
        graph = [[] for _ in range(n)]
        visited = {0}
        for e in connections:
            adjacent_list[e[0]].append(e[1])
            graph[e[0]].append(e[1])
            graph[e[1]].append(e[0])
        def dfs(cur: int):
            for neighbour in graph[cur]:
                if neighbour not in visited:
                    if cur not in adjacent_list[neighbour]:
                        changes[0] += 1
                    visited.add(neighbour)
                    dfs(neighbour)
        dfs(0)
        return changes[0]

    # # 399. Evaluate Division
    # def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    #     dictionary = set()
    #     for e in equations:
    #         if e[0] not in dictionary:
    #             dictionary.add(e[0])
    #         if e[1] not in dictionary:
    #             dictionary.add(e[1])
    #     for q in queries:
    #         cur = 1
    #         for char in q[0]:

    # 1926. Nearest Exit from Entrance in Maze
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        # def isInBound(index: int, bound: int):
        #     return 0 <= index < bound
        #
        # def bfs(level: int):
        #     while q:
        #         cur = q.pop(0)
        #         index = cur[0] * len(maze[0]) + cur[1]
        #         for node in adj_list[index]:
        #             if node not in visited:
        #                 if (node[0] == 0 or node[0] == len(maze) - 1 or node[1] == 0 or node[1] == len(maze[0]) - 1) and node != entrance:
        #                     return level
        #                 q.append(node)
        #                 visited.append(node)
        #         bfs(level + 1)
        #
        # adj_list = []
        # for i in range(len(maze)):
        #     for j in range(len(maze[0])):
        #         adj_list.append([])
        #         index = i * len(maze[0]) + j
        #         if isInBound(i + 1, len(maze)) and maze[i + 1][j] == '.':
        #             adj_list[index].append([i + 1, j])
        #         if isInBound(i - 1, len(maze)) and maze[i - 1][j] == '.':
        #             adj_list[index].append([i - 1, j])
        #         if isInBound(j + 1, len(maze[0])) and maze[i][j + 1] == '.':
        #             adj_list[index].append([i, j + 1])
        #         if isInBound(j - 1, len(maze[0])) and maze[i][j - 1] == '.':
        #             adj_list[index].append([i, j - 1])
        # q = [entrance]
        # visited = []
        # bfs(0)
        # return -1

        def isInBound(input: int, bound: int):
            return 0 <= input < bound

        adj_list = []
        for i in range(len(maze)):
            for j in range(len(maze[0])):
                adj_list.append([])
                if maze[i][j] == '.':
                    index = i * len(maze[0]) + j
                    if isInBound(i + 1, len(maze)) and maze[i + 1][j] == '.':
                        adj_list[index].append([i + 1, j])
                    if isInBound(i - 1, len(maze)) and maze[i - 1][j] == '.':
                        adj_list[index].append([i - 1, j])
                    if isInBound(j + 1, len(maze[0])) and maze[i][j + 1] == '.':
                        adj_list[index].append([i, j + 1])
                    if isInBound(j - 1, len(maze[0])) and maze[i][j - 1] == '.':
                        adj_list[index].append([i, j - 1])

        visited = {entrance[0] * len(maze[0]) + entrance[1]: 0}
        q = [entrance]
        while q:
            cur = q.pop(0)
            # print(f"current: {cur}")
            # print(f"adjacents: {adj_list[cur[0] * len(maze[0]) + cur[1]]}")
            cur_index = cur[0] * len(maze[0]) + cur[1]
            for node in adj_list[cur_index]:
                node_index = node[0] * len(maze[0]) + node[1]
                if node_index not in visited.keys():
                    # print(f"node: {node}")
                    if node != entrance and (node[0] == 0 or node[0] == len(maze) - 1 or node[1] == 0 or node[1] == len(maze[0]) - 1):
                        return visited[cur_index] + 1
                    q.append(node)
                    visited[node_index] = visited[cur_index] + 1
            # print("------------------")
        return -1

    # 994. Rotting Oranges
    def orangesRotting(self, grid: List[List[int]]) -> int:
        def isInBound(input: int, bound: int):
            return 0 <= input < bound

        # print(grid, len(grid))
        q = []
        visited = {}
        rotten_ct, fresh_ct = 0, 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                # print(grid[i][j])
                if grid[i][j] == 1:
                    fresh_ct += 1
                elif grid[i][j] == 2:
                    q.append([i, j])
                    visited[i * len(grid[0]) + j] = 0
                    rotten_ct += 1
        if rotten_ct == 0:
            if fresh_ct == 0:
                return 0
            else:
                return -1
        # print(f"number of rotten: {rotten_ct}")
        # print(f"number of fresh: {fresh_ct}")

        dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        level = 0
        covered = 0
        while q:
            # print(f"grid: {grid}")
            cur = q.pop(0)
            cur_index = cur[0] * len(grid[0]) + cur[1]
            # print(f"cur_index: {cur_index}")

            for d in dirs:
                indices = [cur[0] + d[0], cur[1] + d[1]]
                index = indices[0] * len(grid[0]) + indices[1]
                if (isInBound(indices[0], len(grid)) and
                    isInBound(indices[1], len(grid[0])) and
                    grid[indices[0]][indices[1]] == 1 and
                    index not in visited):

                    # print(f"neighbour_index: {index, indices}")
                    grid[indices[0]][indices[1]] = 2
                    q.append(indices)
                    level = visited[cur_index] + 1
                    visited[index]= level
                    covered += 1
        #     print(f"level: {level}")
        #     print("---------------------------------------------------")
        # print(f"fresh: {fresh_ct}, covered: {covered}")
        return level if covered == fresh_ct else -1

    # 215. Kth Largest Element in an Array
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = nums[: k]
        heapq.heapify(heap)
        for num in nums[k: ]:
            if num > heap[0]:
                heapq.heappop(heap)
                heapq.heappush(heap, num)
        return heap[0]

    # 2542. Maximum Subsequence Score
    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        pairs = [[n1, n2] for n1, n2 in zip(nums1, nums2)]
        pairs = sorted(pairs, key=lambda p: p[1], reverse=True)
        res = 0
        n1Sum = 0
        minHeap = []
        for n1, n2 in pairs:
            n1Sum += n1
            heapq.heappush(minHeap, n1)
            if len(minHeap) > k:
                n1Pop = heapq.heappop(minHeap)
                n1Sum -= n1Pop
            if len(minHeap) == k:
                res = max(n1Sum * n2, res)
        return res

    # 2462. Total Cost to Hire K Workers
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        priority_queue = []
        n = len(costs)
        left_candidate_bound = candidates - 1
        right_candidate_bound = n - candidates
        for index in range(candidates):
            priority_queue.append((costs[index], index))
        for index in range(n - candidates, n):
            if index > left_candidate_bound:
                priority_queue.append((costs[index], index))
        heapq.heapify(priority_queue)
        total_cost = 0
        for _ in range(k):
            cost, index = heapq.heappop(priority_queue)
            total_cost += cost
            if index <= left_candidate_bound:
                left_candidate_bound += 1
                if left_candidate_bound < right_candidate_bound:
                    heapq.heappush(priority_queue, (costs[left_candidate_bound], left_candidate_bound))
            if index >= right_candidate_bound:
                right_candidate_bound -= 1
                if left_candidate_bound < right_candidate_bound:
                    heapq.heappush(priority_queue, (costs[right_candidate_bound], right_candidate_bound))
        return total_cost

    # 2300. Successful Pairs of Spells and Potions
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        potions.sort()
        pairs = []
        for s in spells:
            l, h = 0, len(potions) - 1
            while l <= h:
                m = l + int((h - l) / 2)
                if s * potions[m] >= success:
                    h = m - 1
                else:
                    l = m + 1
            pairs.append(len(potions) - l)
        return pairs

    # 162. Find Peak Element
    def findPeakElement(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            m = (l + r) // 2
            if nums[m] > nums[m + 1]:
                r = m
            else:
                l = m + 1
        return l
        # if len(nums) == 2:
        #     return 0 if nums[0] > nums[1] else 1
        # for i in range(1, len(nums) - 1):
        #     if nums[i] > nums[i - 1] and  nums[i] > nums[i + 1]:
        #         return i
        # if len(nums) >= 2 and nums[-1] > nums[-2]:
        #     return len(nums) - 1
        # return 0

    # 875. Koko Eating Bananas
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        l, r = 1, max(piles)
        if len(piles) == 1:
            return math.ceil(piles[0] / h)
        while l < r:
            bananas = piles.copy()
            m = (l + r) // 2
            ct = 0
            print(f"l, m, h: {l, m, r}")
            while bananas:
                print(bananas)
                if bananas[0] > m:
                    ct += math.ceil(bananas[0] / m)
                else:
                    ct += 1
                bananas.pop(0)
            print(f"ct: {ct}")
            if ct > h:
                l = m + 1
            else:
                r = m
        return l

    # 17. Letter Combinations of a Phone Number
    def letterCombinations(self, digits: str) -> List[str]:
        keyboard = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }
        if len(digits) == 0:
            return []
        res = []
        first = digits[0]
        digits = digits[1:]
        for l in keyboard[first]:
            res.append(l)
        for d in digits:
            length = len(res)
            for i in range(length):
                for l in keyboard[d]:
                    res.append(res[i] + l)
                    print(res[i] + l)
            res = res[length:]
        return res

    # 216. Combination Sum III
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        def helper(num, cur, cur_sum):
            # print(round_left)
            if len(cur) == k:
                if cur_sum == n:
                    res.append(cur)
            for c in range(num, max_num + 1):
                subtotal = cur_sum + c
                if subtotal <= n:
                    helper(c + 1, cur + [c], subtotal)
                # else:
                #     return
        if k > n:
            return []
        max_num = 0
        for i in range(1, k):
            max_num += i
        max_num = n - max_num
        # res = [[i] for i in range(1, max_num + 1)]
        res = []
        helper(1, [], 0)
        return res

    # 1137. N-th Tribonacci Number
    def tribonacci(self, n: int) -> int:
        if n == 0:
            return 0
        elif n < 3:
            return 1
        fib = [0] * (n + 1)
        fib[1], fib[2] = 1, 1
        print(fib)
        for i in range(3, n + 1):
            fib[i] = fib[i - 1] + fib[i - 2] + fib[i - 3]
        return fib[-1]

    # 746. Min Cost Climbing Stairs
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        length = len(cost)
        dp = [0] * length
        dp[0] = cost[0]
        dp[1] = cost[1]
        for i in range(2, length):
            # print(dp)
            # if dp[i - 2] < dp[i - 1]:
            #     print(f"at {i}th step, choose {i - 2}, cost {cost[i - 2]}")
            #     dp[i] = dp[i - 2] + cost[i]
            # else:
            #     print(f"at {i}th step, choose {i - 1}, cost {cost[i - 1]}")
            #     dp[i] = dp[i - 1] + cost[i]
            dp[i] = min(dp[i - 2], dp[i - 1]) + cost[i]
        return min(dp[-1], dp[-2])

    # 198. House Robber
    def rob(self, nums: List[int]) -> int:
        length = len(nums)
        if length < 2:
            return nums[0]
        dp = [0] * length
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, length):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
        return dp[-1]

    # 62. Unique Paths
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n] * m
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    # 1143. Longest Common Subsequence
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        l1, l2 = len(text1) + 1, len(text2) + 1
        # dp = [[0] * l2] * l1  # all lines have the same address
        dp = [[0] * l2 for _ in range(l1)]
        for i in range(1, l1):
            for j in range(1, l2):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = max(dp[i - 1][j - 1] + 1, dp[i - 1][j], dp[i][j - 1])
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    # 121. Best Time to Buy and Sell Stock
    def maxProfit(self, prices: List[int]) -> int:
        length = len(prices)
        dp = [0] * length
        cur_min = 1e4
        for i in range(length):
            if prices[i] < cur_min:
                cur_min = prices[i]
            else:
                dp[i] = prices[i] - cur_min
        return max(dp)

    # 714. Best Time to Buy and Sell Stock with Transaction Fee
