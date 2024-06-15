class MessageTask:
    def __init__(self):
        self.msgType = 0
        self.usrInst = 0
        self.exeTime = 0
        self.deadLine = 0
        self.startTime = 0


MAX_USER_ID = 10005


def main():
    # 1. 读取任务数、核数、系统最大执行时间
    n, m, c = map(int, input().split())

    # 2. 读取每个任务的信息
    tasks = []
    for _ in range(n):
        msgType, usrInst, exeTime, deadLine = map(int, input().split())
        deadLine = min(deadLine, c)
        task = MessageTask()
        task.msgType, task.usrInst, task.exeTime, task.deadLine = msgType, usrInst, exeTime, deadLine
        tasks.append(task)

    # 按任务截止时间排序
    tasks.sort(key=lambda x: x.deadLine)

    # 初始化核状态
    cores = [[] for _ in range(m)]
    core_end_times = [0] * m  # 记录每个核的结束时间
    user_core_assignment = [-1] * MAX_USER_ID  # 记录每个用户的核分配

    # 3. 调度逻辑：按截止时间排序并调度
    user_last_task_end = [-1] * MAX_USER_ID  # 记录每个用户上一个任务的结束时间

    for task in tasks:
        uid = task.usrInst

        if user_core_assignment[uid] == -1:  # 如果是该用户的第一个任务
            candidate_cores = [(core_end_times[core_id], core_id) for core_id in range(m) if
                               core_end_times[core_id] + task.exeTime <= task.deadLine]
            if candidate_cores:
                min_end_time, min_core_id = min(candidate_cores)
                task.startTime = core_end_times[min_core_id]
                cores[min_core_id].append(task)
                core_end_times[min_core_id] += task.exeTime
                user_last_task_end[uid] = core_end_times[min_core_id]
                user_core_assignment[uid] = min_core_id
        else:
            core_id = user_core_assignment[uid]
            if core_end_times[core_id] >= user_last_task_end[uid] and core_end_times[
                core_id] + task.exeTime <= task.deadLine:
                task.startTime = core_end_times[core_id]
                cores[core_id].append(task)
                core_end_times[core_id] += task.exeTime
                user_last_task_end[uid] = core_end_times[core_id]
            else:
                task.startTime = max(core_end_times[core_id], user_last_task_end[uid])
                cores[core_id].append(task)
                core_end_times[core_id] = task.startTime + task.exeTime
                user_last_task_end[uid] = core_end_times[core_id]

    # 4. 对每个核心内的任务进行调整以提高亲和性评分
    for core_id in range(m):
        core_tasks = cores[core_id]
        if not core_tasks:
            continue

        # 使用贪心策略调整核心内的任务顺序以提高亲和性评分
        core_tasks.sort(key=lambda task: (task.msgType, task.startTime))

    # 5. 输出结果
    output_lines = []
    for coreId, core_tasks in enumerate(cores):
        line = str(len(core_tasks))
        for task in core_tasks:
            line += f" {task.msgType} {task.usrInst}"
        output_lines.append(line + "\n")

    print(''.join(output_lines), end='')


if __name__ == "__main__":
    main()
