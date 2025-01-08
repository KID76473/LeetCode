class Solution_oneline_assessment:
    # Tiktok 11.30 oa
    def interesting_watch_sequence(self, v):
        res = 0
        i = 0
        # print(len(v))
        while i < len(v) - 1:
            print(i, v[i])
            if v[i] >= v[i + 1]:
                # i -= 1
                j = i - 1
                res += 1
                while j >= 0:
                    print(j)
                    if v[j] > v[i + 1]:
                        v.pop(j)
                        res += 1
                        j -= 1
                    else:
                        break
                    print(f"res:{res}")
                v.pop(i)
            else:
                i += 1
        return res - 1

    # Tiktok 12.11 OA
    # Given an array of delay of server and maximal number of server to jump over k, give an algorithm with minimal delay
    def getMinimumUploadDelay(self, delay, k):
        length = len(delay)
        dp = [0] * length
        for i in range(k):
            dp[i] = delay[i]
        for i in range(k, length):
            dp[i] = min(dp[i - k, i]) + delay[i]
        return dp[-1]

    # IBM 12.17 OA
    def countValidSubstrings(self, s, minLen, maxLen):
        def count(last_index, cur_index):
            result = 0
            for i in range(minLen, min(maxLen + 1, n)):
                result += cur_index + 1 - last_index - i + 1
            return result

        res = 0
        n = len(s)
        if minLen == 1:
            res += n
        last = 0
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                res += count(last, i)
                last = i + 1
        res += count(last, n - 1)
        return res
