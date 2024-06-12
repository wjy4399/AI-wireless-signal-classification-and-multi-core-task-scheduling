import math

class MessageTask:
    def __init__(self, msgType, usrInst, exeTime, deadLine, idx):
        self.msgType = msgType
        self.usrInst = usrInst
        self.exeTime = exeTime
        self.deadLine = deadLine
        self.idx = idx
        self.endTime = 0

MAX_USER_ID = 10005
ACCURACY = 0.69

def adjust_execution_time(exeTime, accuracy):
    adjusted_time = exeTime * (2 - accuracy * accuracy)
    return round(adjusted_time)

def calculate_scores(cores):
    affinity_score = 0
    completed_tasks = 0

    for core in cores:
        if not core:
            continue

        for i in range(1, len(core)):
            if core[i].msgType == core[i - 1].msgType:
                affinity_score += 1

        for task in core:
            if task.endTime <= task.deadLine:
                completed_tasks += 1

    return affinity_score, completed_tasks

def calculate_normalized_score(affinity_score, completed_tasks, N):
    absolute_upper_bound = N + N
    raw_score = affinity_score + completed_tasks
    normalized_score = 100000 * math.log10(N) * (1 + (raw_score - absolute_upper_bound) / absolute_upper_bound)
    return normalized_score

def greedy_schedule(tasks, m):
    cores = [[] for _ in range(m)]
    core_end_times = [0] * m

    for uid_tasks in tasks:
        if not uid_tasks:
            continue

        # 找到在不超时情况下亲和力最高的核
        best_core = -1
        max_affinity = -1
        min_end_time = float('inf')
        for i in range(m):
            current_affinity = 0
            if cores[i] and cores[i][-1].msgType == uid_tasks[0].msgType:
                current_affinity = 1

            if core_end_times[i] + sum(task.exeTime for task in uid_tasks) <= min(uid_tasks, key=lambda x: x.deadLine).deadLine:
                if current_affinity > max_affinity or (
                        current_affinity == max_affinity and core_end_times[i] < min_end_time):
                    max_affinity = current_affinity
                    min_end_time = core_end_times[i]
                    best_core = i

        # 如果没有找到满足条件的核，选择当前负载最小的核
        if best_core == -1:
            best_core = core_end_times.index(min(core_end_times))

        for task in uid_tasks:
            task.endTime = core_end_times[best_core] + task.exeTime
            cores[best_core].append(task)
            core_end_times[best_core] += task.exeTime

    return cores

def main():
    n, m, c = map(int, input().split())
    tasks = [[] for _ in range(MAX_USER_ID)]

    for idx in range(n):
        msgType, usrInst, exeTime, deadLine = map(int, input().split())
        exeTime = adjust_execution_time(exeTime, ACCURACY)
        deadLine = min(deadLine, c)
        task = MessageTask(msgType, usrInst, exeTime, deadLine, idx)
        tasks[usrInst].append(task)

    for uid in range(MAX_USER_ID):
        tasks[uid].sort(key=lambda x: (x.deadLine, x.exeTime))

    cores = greedy_schedule(tasks, m)

    output_lines = []
    for coreId, core_tasks in enumerate(cores):
        core_tasks.sort(key=lambda x: x.idx)
        line = f"{len(core_tasks)}"
        for task in core_tasks:
            line += f" {task.msgType} {task.usrInst}"
        output_lines.append(line)

    print('\n'.join(output_lines), end='')

if __name__ == "__main__":
    main()
