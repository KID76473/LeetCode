from typing import List


class Solution:
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

