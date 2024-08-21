import collections
import math
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
