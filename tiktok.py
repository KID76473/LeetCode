def newspaper(paragraphs, aligns, width):
    def helper(content, left):
        res = ''
        num = width - len(content)
        if left:
            res += content + '-' * num
        else:
            res += '-' * num + content
        return '*' + res + '*'
    all_stars = '*' * (width + 2)
    res = [all_stars]
    for i in range(len(paragraphs)):
        print(f'paragraph: {paragraphs[i]}')
        length = 0
        words = ''
        arr = paragraphs[i]
        while arr:
            word = arr[0]
            add_len = len(word) if len(words) == 0 else len(word) + 1
            if length + add_len > width:
                print(helper(words, aligns[i]))
                res.append(helper(words, aligns[i]))
                length = 0
                words = ''
            else:
                words += ' ' if words != '' else ''
                words += word
                length += add_len
                arr.remove(word)
        if length != 0:
            print(helper(words, aligns[i]))
            res.append(helper(words, aligns[i]))
    res.append(all_stars)
    return res

# a = [['hello', 'world'], ['how', 'areyou', 'doing'], ['please look', 'and align', 'to right']]
# b = [True, False, False]
# c = 16

a = [["a", "bb", "cc"]]
b = [True]
c = 5

# a = [["hello"]]
# b = ["LEFT"]
# c = 5

# result = newspaper(a, b, c)
# print('---------------------------------------------------------------')
# for x in result:
#     print(x, len(x))

def structure(arr):
    n = len(arr)
    max_val = max([arr[i] - i for i in range(n)])
    print(f'max: {max_val}')
    res1 = 0
    for i in range(n):
        num = max_val - arr[i] + i
        res1 += num
        print(i, num)
    print('--------------------')
    max_val = max([arr[i] - (n - 1 - i) for i in range(n)])
    print(f'max: {max_val}')
    res2 = 0
    for i in range(n - 1, -1, -1):
        num = max_val - arr[i] + n - 1 - i
        res2 += num
        print(i, num)
    print('--------------------')
    print(res1, res2)
    return min(res1, res2)


# a = [1, 4, 3, 2]
# a = [5, 1, 1, 1]
# a = [9, 7, 7, 7, 8]
# print(structure(a))


# def figures(shapes, n, m):
#     def draw(shape, x, y, idx):
#         if shape == 'A':
#             if grid[x][y] == 0:
#                 grid[x][y] = idx
#                 return True
#             else:
#                 return False
#         elif shape == 'B':
#             if y + 2 >= m:
#                 return False
#             elif grid[x][y] == 0 and grid[x][y + 1] == 0 and grid[x][y + 2] == 0:
#                 grid[x][y] = idx
#                 grid[x][y + 1] = idx
#                 grid[x][y + 2] = idx
#                 return True
#             else:
#                 return False
#         elif shape == 'C':
#             # print(grid, x, y)
#             if x + 1 >= n or y + 1 >= m:
#                 return False
#             elif grid[x][y] == 0 and grid[x][y + 1] == 0 and grid[x + 1][y] == 0 and grid[x + 1][y + 1] == 0:
#                 grid[x][y] = idx
#                 grid[x][y + 1] = idx
#                 grid[x + 1][y] = idx
#                 grid[x + 1][y + 1] = idx
#                 return True
#             else:
#                 return False
#         elif shape == 'D':
#             # print(f'D: {x + 2, n}, {y + 1, m}')
#             print(f'D: {grid}')
#             if x + 2 >= n or y + 1 >= m:
#                 return False
#             elif grid[x][y] == 0 and grid[x + 1][y + 1] == 0 and grid[x + 1][y] == 0 and grid[x + 2][y] == 0:
#                 grid[x][y] = idx
#                 grid[x + 1][y + 1] = idx
#                 grid[x + 1][y] = idx
#                 grid[x + 2][y] = idx
#                 print(f'D: {grid}')
#                 return True
#             else:
#                 return False
#         else: # shape == 'E'
#             if x + 1 >= n or y + 2 >= m:
#                 return False
#             elif grid[x + 1][y] == 0 and grid[x + 1][y + 1] == 0 and grid[x + 1][y + 2] == 0 and grid[x][y + 1] == 0:
#                 grid[x + 1][y] = idx
#                 grid[x + 1][y + 1] = idx
#                 grid[x + 1][y + 2] = idx
#                 grid[x][y + 1] = idx
#                 return True
#             else:
#                 return False
#     grid = [[0] * m for _ in range(n)]
#     for idx, s in enumerate(shapes):
#         for i in range(n):
#             stop = False
#             for j in range(m):
#                 if draw(s, i, j, idx + 1):
#                     stop = True
#                     break
#             if stop:
#                 break
#         print(idx + 1, s, grid)
#     return grid


def figures(shapes, n, m):
    def draw(shape, x, y, idx):
        if shape == 'A':
            if grid[x][y] == 0:
                grid[x][y] = idx
                return True
            else:
                return False
        elif shape == 'B':
            pts = [(x, y), (x, y + 1), (x, y + 2)]
            if y + 2 >= m:
                return False
            elif all(grid[a][b] == 0 for a, b in pts):
                for a, b in pts:
                    grid[a][b] = idx
                return True
            else:
                return False
        elif shape == 'C':
            pts = [(x, y), (x, y + 1), (x + 1, y), (x + 1, y + 1)]
            if x + 1 >= n or y + 1 >= m:
                return False
            elif all(grid[a][b] == 0 for a, b in pts):
                for a, b in pts:
                    grid[a][b] = idx
                return True
            else:
                return False
        elif shape == 'D':
            pts = [(x, y), (x + 1, y + 1), (x + 1, y), (x + 2, y)]
            if x + 2 >= n or y + 1 >= m:
                return False
            elif all(grid[a][b] == 0 for a, b in pts):
                for a, b in pts:
                    grid[a][b] = idx
                return True
            else:
                return False
        else: # shape == 'E'
            pts = [(x, y + 1), (x + 1, y), (x + 1, y + 1), (x + 1, y + 2)]
            if x + 1 >= n or y + 2 >= m:
                return False
            elif all(grid[a][b] == 0 for a, b in pts):
                for a, b in pts:
                    grid[a][b] = idx
                return True
            else:
                return False
    grid = [[0] * m for _ in range(n)]
    for idx, s in enumerate(shapes):
        for i in range(n):
            stop = False
            for j in range(m):
                if draw(s, i, j, idx + 1):
                    stop = True
                    break
            if stop:
                break
        print(idx + 1, s, grid)
    return grid

# a = ['D', 'B', 'A', 'C']
# b = 4
# c = 4
# print(figures(a, b, c))


class memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.mem = [0] * capacity
        self.idx = []
        self.ct = 0

    def alloc(self, length):
        start = 0
        while start * 8 < len(self.mem):
            if all(x == 0 for x in self.mem[start * 8: start * 8 + length]) and start * 8 + length <= self.capacity:
                for i in range(length):
                    self.mem[start * 8 + i] = self.ct
                self.idx.append([self.ct, start * 8, length])
                self.ct += 1
                return start * 8
            start += 1
        return -1

    def erase(self, id):
        for i in range(len(self.idx)):
            if self.idx[i][0] == id:
                for j in range(self.idx[i][2]):
                    self.mem[self.idx[i][1] + j] = 0
                ret = self.idx[i][2]
                self.idx.pop(i)
                return ret
        return -1

# # This takes O(n^2)
# def mountains(heights, viewingGap):
#     def helper(idx):
#         i = idx - viewingGap
#         res = max(heights)
#         cur = heights[idx]
#         while i >= 0:
#             res = min(abs(cur - heights[i]), res)
#             i -= 1
#         i = idx + viewingGap
#         while i < len(heights):
#             print(i, heights[i], abs(cur - heights[i]))
#             res = min(abs(cur - heights[i]), res)
#             i += 1
#         return res
#     result = max(heights)
#     for i in range(len(heights)):
#         print('---------------')
#         print(f'result {i}: {helper(i)}')
#         result = min(result, helper(i))
#     return result

import bisect
def mountains(heights, viewingGap):
    n = len(heights)
    def binary(target, arr):
        length = len(arr)
        mid = length // 2
        low, high = 0, length - 1
        while 0 <= mid < length:
            if arr[mid] > target:
                high = mid
            elif arr[mid] < target:
                low = mid
            else:
                return arr[mid]
            mid = low + (high - low) // 2
        return arr[mid]
    res = max(heights)
    for i in range(n):
        temp = min(abs(heights[i] - heights[bisect.bisect_left(heights[i + viewingGap + 1: ], heights[i])]),
                   abs(heights[i] - heights[bisect.bisect_left(heights[: i - viewingGap], heights[i])]))
        res = min(temp, res)
        # res = min(abs(heights[i] - binary(heights[i], heights[i + viewingGap: ])),
        #           abs(heights[i] - binary(heights[i], heights[i - viewingGap: ])))
    return res


# a = [4, 2, 5, 9, 4, 6, 8]
# b = 2
# a = [1, 5, 4, 10, 9]
# b = 3
# print(mountains(a, b))
# print('------------------------')


def binary(arr, target):
    l, h = 0, len(arr) - 1
    while l <= h:
        m = (l + h) // 2
        if arr[m] == target:
            return m
        elif arr[m] > target:
            h = m - 1
        else:
            l = m + 1
    return l


a = [0, 3, 6, 9, 12]
print(binary(a, 2))
# print(bisect.bisect_left(a, 1))


def veowls(word):
    if len(word) < 3:
        return -1
    v = {'a', 'e', 'i', 'o', 'u'}
    res = 0
    cur = 0
    for c in word[: 3]:
        if c in v:
            cur += 1
    if cur == 2:
        res += 1
    for i in range(len(word[3: ])):
        if word[i] in v:
            cur += 1
        if word[i - 3] in v:
            cur -= 1
        if cur == 2:
            res += 1
    return res


def drones(stations, target):
    stations = sorted(stations)
    stations.insert(0, 0)
    stations.append(target)
    n = len(stations)
    res = 0
    for i in range(1, n):
        if stations[i] - stations[i - 1] > 10:
            res += stations[i] - stations[i - 1] - 10
    return res


def x_shape(matrix):
    def find(x, y):
        ways = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        i = 1
        res = 1
        print(f'x: {x}, y: {y}')
        while 0 <= x - i and x + i <= n - 1 and 0 <= y - i and y + i <= m - 1:
            print(i)
            for way in ways:
                if matrix[x + way[0] * i][y + way[1] * i] == 0:
                    return res
            res += 1
            i += 1
        return res
    n, m = len(matrix), len(matrix[0])
    length = 0
    result = [-1, -1]
    for j in range(m):
        for i in range(n):
            if matrix[i][j] == 1:
                cur = find(i, j)
                if cur > length:
                    length = cur
                    result = [i, j]
    print(f'final length: {length}')
    return  result


# a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
# a = [[1, 1], [1, 1]]
# print(x_shape(a))


def obstacles(operations):
    res = ''
    obs = []
    for cmd in operations:
        op, x = cmd[: 2]
        # print(cmd, op, x)
        print(obs)
        if op == 1:
            bisect.insort(obs, x)  # equals to following two
            # idx = bisect.bisect_left(obs, x)
            # obs.insert(idx, x)
        elif op == 2:
            size = cmd[2]
            # following equals to comments below
            left = bisect.bisect_left(obs, x - size)
            right = bisect.bisect_left(obs, x)
            res += '0' if left < right else '1'
            # idx = bisect.bisect_left(obs, x)
            # print(f'idx: {idx}')
            # if idx == 0:
            #     res += '1'
            # elif obs[idx - 1] >= x - size:
            #     res += '0'
            # else:
            #     res += '1'
        else:
            print('wrong!!!!!!!!!!')
        print(f'res: {res}')
    return res

# a = [[1, 2], [1, 5], [2, 5, 2], [2, 6, 3], [2, 2, 1], [2, 3, 2]]
# a = [[1, 3], [2, 3, 2], [2, 4, 1]]
# print(obstacles(a))


def swap(numbers):
    def helper(num):
        for i in range(n):
            for j in range(i + 1, n):
                num_string = str(num)
                temp = num_string[i]
                num_string[i] = num_string[j]
                num_string[j] = temp
                all_nums.add(int(num_string))
        return all_nums
    res = 0
    n = len(numbers)
    all_nums = {}
    for num in range(n):
        helper(num)
    for num in range(n):
        if num in all_nums:
            res += 1
    return res

# 给定一个正整数数组 nums 和一个目标值k，断是否存在一个连续子数组，其元素之和恰好等于k
def subarray1(nums, k):
    cur = nums[0]
    n = len(nums)
    for i in range(n):
        j = i
        while j < n and cur < k:
            cur += nums[j]
            j += 1
        if cur > k:
            cur -= nums[i]
        else:
            return True
    return False

a = [1, 3, 4, 2, 4]
b = 10
print(subarray1(a, b))
