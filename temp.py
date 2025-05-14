def sol(n1, n2):
    m1, m2 = {}, {}
    for x in n1:
        # print(x, y)
        if x not in m1:
            m1[x] = 1
        else:
            m1[x] += 1
    for y in n2:
        if y not in m2:
            m2[y] = 1
        else:
            m2[y] += 1
    # print(m1, m2)
    res = []
    for x in m1:
        if x in m2:
            for _ in range(min(m1[x], m2[x])):
                res.append(x)
    return res

nums1 = [1, 3, 5]
nums2 = [1]
print(sol(nums1, nums2))
# print('------------------------------')

nums1 = [1, 2, 3, 4, 7, 8, 9, 3, 2, 1]
nums2 = [2, 2, 1, 6]
print(sol(nums1, nums2))
# print('------------------------------')

nums1 = [9, 8, 7, 6, 5, 1, 1, 1, 2, 2, 3]
nums2 = [10, 8, 1, 2, 3]
print(sol(nums1, nums2))