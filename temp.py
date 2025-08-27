import bisect

a = [1, 3, 5]
print(bisect.bisect(a, 3))
print(bisect.bisect_left(a, 3))
print(bisect.bisect_right(a, 3))
