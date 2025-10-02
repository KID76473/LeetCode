#
#
# def find(target, arr):
#     l, h = 0, len(arr) - 1
#     while l <= h:
#         m = (l + h) // 2
#         print(l, m, h)
#         if arr[m] == target:
#             return m
#         elif arr[m] < target:
#             l = m + 1
#         else:
#             h = m - 1
#     return l
#
# a = [1, 3, 4, 6, 9, 12, 13]
# idx = find(10, a)
# print(idx, a[idx])

# a = 'ab'
# c = 'cde'
# for char1, char2 in zip(a, c):
#     print(char1, char2)


# # {<>}<>{}{<><>}
# # {}{<>}{}
# def sol(l, m, n):
#     def helper(c, x, index):
#         print(f'before: {c}')
#         if len(c) == 0:
#             return x
#         if c[index] != x[1]:  # cannot be <<>>
#             c = c[: index] + x + c[index: ]
#         print(f'after: {c}')
#         return c
#     res = []
#     start = ''
#     for _ in range(n):
#         start += '{}'
#     res.append(start)
#     print(res, len(res))
#     # <>
#     idx, ct = 0, len(res)
#     while idx < ct:
#         print(f'<>: {idx}')
#         temp = res.pop(0)
#         cur = temp
#         for _ in range(m):   # insert m <>
#             for j in range(len(cur)):
#                 temp = helper(temp, "<>", j)
#                 if temp not in res:
#                     res.append(temp)
#                 temp = cur
#         idx += 1
#     # ()
#     idx, ct = 0, len(res)
#     while idx < ct:
#         print(f'(): {idx}')
#         temp = res.pop(0)
#         cur = temp
#         for _ in range(l):  # insert l ()
#             for j in range(len(cur)):
#                 temp = helper(temp, "()", j)
#                 if temp not in res:
#                     res.append(temp)
#                 temp = cur
#         idx += 1
#     return list(res)
#
# print(sol(1, 1, 1))

a = [[0, 'afdfs'], [3, 'bw3r3'], [6, '2qwf2']]
import bisect
print(bisect.bisect(a, 1))
