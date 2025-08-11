# Recursive Digit Sum
def superDigit(n, k):
    # # mehotd 1
    # x = sum(map(int, n)) * k
    # # digital root: 1 + (x-1) % 9, except that x==0 should map to 0
    # return 1 + (x - 1) % 9 if x else 0

    # method 2
    total = sum(int(d) for d in n) * k

    while total >= 10:
        total = sum(int(d) for d in str(total))
    return total

    # # my method (not work, hitting performance or memory limit)
    # first = True
    # n = int(n)
    # while n >= 10:
    #     # print(n)
    #     temp = n
    #     n = 0
    #     # print(f'temp: {temp}')
    #     while temp > 0:
    #         # print(temp)
    #         n += temp % 10
    #         temp //= 10
    #     if first:
    #         n *= k
    #         first = False
    # return n


# Queue using Two Stacks
def minimumBribes(q):
    n = len(q)
    # 1) check for chaos
    for i, P in enumerate(q):
        if P - (i+1) > 2:
            print("Too chaotic")
            return

    # 2) count total bribes (inversions) in the limited window
    bribes = 0
    for i, P in enumerate(q):
        # only people who started at most two places behind P could have bribed P
        for j in range(max(0, P-2), i):
            if q[j] > P:
                bribes += 1
    print(bribes)


# Queue using Two Stacks
import sys


def dequeue():
    if not s_out:
        while s_in:
            s_out.append(s_in.pop())
    if not s_out:
        print('no value')
    else:
        s_out.pop()


def peek():
    if not s_out:
        while s_in:
            s_out.append(s_in.pop())
    if not s_out:
        print('no value')
    else:
        print(s_out[-1])


s_in, s_out = [], []
for line in sys.stdin:
    cmd = line.strip().split()
    if cmd[0] == '1':
        s_in.append(cmd[1])
    elif cmd[0] == '2':
        dequeue()
    elif cmd[0] == '3':
        peek()
