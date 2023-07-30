s = 'babab'
maxlen = 1
if len(s) < 2:
    result = s

# dynamic programming #################################################################################################
begin = 0
end = 0
# initialize
dp = []
for i in range(len(s)):
    dp.append([])
    for j in range(len(s)):
        dp[i].append([])
        if i == j:
            dp[i][j] = True
# fill in the form
for j in range(1, len(s)):
    for i in range(0, j):
        if s[i] == s[j] and (dp[i + 1][j - 1] or j - i < 3):
            dp[i][j] = True
            temp = j - i + 1
            if maxlen < j - i + 1:
                maxlen = temp
                begin = i
                end = j
        else:
            dp[i][j] = False
result = s[begin: end + 1]
print(result, maxlen)
# dynamic programming #################################################################################################

# center based with odd and even #######################################################################################
# def palindrome(ss) -> bool:
#     l = 0
#     r = len(ss) - 1
#     while r - l >= 1:
#         if ss[l] != ss[r]:
#             return False
#         l += 1
#         r -= 1
#     return True
#
#
# result = ''
# # odd
# for i in range(1, len(s)):
#     l = i - 1
#     r = i + 1
#     while l >= 0 and r < len(s) and palindrome(s[l: r + 1]):
#         temp = r - l + 1
#         if temp > maxlen:
#             maxlen = temp
#             result = s[l: r + 1]
#             print("odd called!")
#         l -= 1
#         r += 1
# # even
# for i in range(len(s) - 1):
#     l = i
#     r = i + 1
#     while l >= 0 and r < len(s) and palindrome(s[l: r + 1]):
#         temp = r - l + 1
#         if temp > maxlen:
#             maxlen = temp
#             result = s[l: r + 1]
#             print("even called!")
#         l -= 1
#         r += 1
# if result == '':
#     result = s[0]
# print(result, maxlen)
# center based with odd and even #######################################################################################

# brute force ##########################################################################################################
# begin = 0
# def palindrome(ss) -> bool:
#     l = 0
#     r = len(ss) - 1
#     while r - l >= 1:
#         if ss[l] != ss[r]:
#             return False
#         l += 1
#         r -= 1
#     return True
#
#
# for i in range(len(s) - 1):
#     for j in range(i + 1, len(s)):
#         temp = j - i + 1
#         print(s[i: j + 1])
#         if palindrome(s[i: j + 1]) and temp > maxlen:
#             begin = i
#             maxlen = temp
# print(begin, maxlen)
# result = s[begin: begin + maxlen]
# print(result, maxlen)
# brute force ##########################################################################################################