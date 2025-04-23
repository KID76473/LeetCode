import sys
from collections import deque, defaultdict


class huawei_solution:
    # https://blog.csdn.net/weixin_42433507/article/details/143171185
    def detect_enemy(self, map, threshold):
        def bfs(index1, index2):
            queue = [[index1, index2]]
            visited.append([index1, index2])
            enemy = 0
            while queue:
                cur = queue.pop(0)
                for x, y in [[1, 0], [0, -1], [-1, 0], [0, 1]]:
                    next_i, next_j = cur[0] + x, cur[1] + y
                    if -1 < next_i < m and -1 < next_j < n and [next_i, next_j] not in visited:
                        visited.append([next_i, next_j])
                        print(f'add {next_i, next_j} to visited')
                        if map[next_i][next_j] == 'E':
                            print(f'found enemy at {next_i, next_j}')
                            enemy += 1
            return enemy

        m, n = len(map), len(map[0])
        res = 0
        # queue = [[0, 0]]
        visited = [[0, 0]]
        # add all walls to visited
        for i in range(m):
            for j in range(n):
                if map[i][j] == '#':
                    visited.append([i, j])
        # explore all available area
        for i in range(m):
            for j in range(n):
                if map[i][j] == '#':
                    continue
                elif [i, j] not in visited and bfs(i, j) < threshold:
                    res += 1
        return res


    # https://blog.csdn.net/weixin_42433507/article/details/142285800
    def survival(self, arr):
        res = 0
        index, length = 0, len(arr)
        queue = []
        while index < length:
            cur = arr[index]
            if cur > 0:
                if queue and queue[-1] < 0:
                    queue.append(queue.pop() + cur)
                else:
                    queue.append(cur)
            else:
                if queue and queue[-1] > 0:
                    if cur + queue[-1] != 0:
                        queue.append(queue.pop() + cur)
                    else:
                        queue.pop()
                else:
                    res += 1
            index += 1
            print(queue)
        res += len(queue)
        return res


    def test_cases(self, input_map):
        cases = input_map
        n = len(input_map[0])

        cases_masks = []
        for row in cases:
            mask = 0
            for k in range(n):
                if row[k] == 1:
                    mask |= 1 << k
            cases_masks.append(mask)

        # Check if all modules are covered by at least one case
        for k in range(n):
            module_covered = False
            for mask in cases_masks:
                if (mask & (1 << k)) != 0:
                    module_covered = True
                    break
            if not module_covered:
                print(-1)
                exit()

        current_mask = (1 << n) - 1
        count = 0

        while current_mask != 0:
            best_idx = -1
            best_overlap = 0
            max_cnt = 0
            for idx in range(len(cases_masks)):
                overlap = cases_masks[idx] & current_mask
                cnt = overlap.bit_count()
                if cnt > max_cnt:
                    max_cnt = cnt
                    best_overlap = overlap
                    best_idx = idx
            if best_idx == -1:
                print(-1)
                exit()
            current_mask &= ~best_overlap
            count += 1
        return count
        # def find_min_index():
        #     ind, val = float('inf'), -1
        #     res = []
        #     for i, c in enumerate(cases):
        #         if c < val:
        #             val = c
        #             ind = i
        #             res.append(i)
        #     return res, val
        # def find_unsolved():
        #     for i in range(n):
        #         temp = 0
        #         for j in range(m):
        #             temp += 1
        #         cases[i] = temp
        #     return cases
        #
        # m, n = len(map), len(map[0])
        # cases = [0 for _ in range(n)]
        # unsolved = [i for i in range(n)]
        # res = []
        # find_unsolved()
        # while unsolved:
        #     index_list, value = find_min_index()
        #     for i in index_list:
        #
        # return len(res)


    def subway(self, lines, start, end):
        # n = int(sys.stdin.readline())
        # lines = []
        # for _ in range(n):
        #     stations = sys.stdin.readline().split()
        #     lines.append(stations)
        # start, end = sys.stdin.readline().split()

        # 构建站点到线路的映射
        n = len(lines)
        station_lines = defaultdict(list)
        for i in range(n):
            for s in lines[i]:
                station_lines[s].append(i)
        print(f'station_lines: {station_lines}')

        # 处理无效输入情况
        if start not in station_lines or end not in station_lines:
            print(-1)
            sys.exit()

        # 初始节点和终止节点集合
        start_nodes = [(i, start) for i in station_lines[start]]
        end_nodes = set((i, end) for i in station_lines[end])

        # 初始化距离字典和队列
        distance = defaultdict(lambda: float('inf'))
        for node in start_nodes:
            distance[node] = 0
        queue = deque(start_nodes)
        print(queue)

        # BFS处理
        while queue:
            print(queue)
            u = queue.popleft()
            current_cost = distance[u]

            if u in end_nodes:
                print(f'final price{current_cost + 2}')
                return current_cost + 2

            i, s = u
            line = lines[i]
            try:
                pos = line.index(s)
            except ValueError:
                continue  # 理论上不会发生

            # 处理同一线路的前后站点
            if pos > 0:
                prev_s = line[pos - 1]
                v = (i, prev_s)
                if distance[v] > current_cost:
                    distance[v] = current_cost
                    queue.appendleft(v)
            if pos < len(line) - 1:
                next_s = line[pos + 1]
                v = (i, next_s)
                if distance[v] > current_cost:
                    distance[v] = current_cost
                    queue.appendleft(v)

            # 处理换乘到其他线路
            for j in station_lines[s]:
                if j == i:
                    continue
                v = (j, s)
                new_cost = current_cost + 1
                if distance[v] > new_cost:
                    distance[v] = new_cost
                    queue.append(v)

        return -1
