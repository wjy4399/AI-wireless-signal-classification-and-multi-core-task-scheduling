#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/5/10 下午8:33
# @File    : main.py
# @desc    : 智联杯示例代码
# @license : Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

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

    # 3. 调度逻辑：按截止时间排序并调度
    user_last_task_end = [-1] * MAX_USER_ID  # 记录每个用户上一个任务的结束时间

    for task in tasks:
        uid = task.usrInst

        if user_last_task_end[uid] == -1:  # 如果是该用户的第一个任务
            candidate_cores = [(core_end_times[core_id], core_id) for core_id in range(m) if
                               core_end_times[core_id] + task.exeTime <= task.deadLine]
            if candidate_cores:
                min_end_time, min_core_id = min(candidate_cores)
                task.startTime = core_end_times[min_core_id]
                cores[min_core_id].append(task)
                core_end_times[min_core_id] += task.exeTime
                user_last_task_end[uid] = core_end_times[min_core_id]
        else:
            inserted = False
            # 查找从上一个任务结束时间点后的任务匹配，包括所有核
            candidate_cores = [(core_end_times[core_id], len(cores[core_id]), core_id) for core_id in range(m) if
                               core_end_times[core_id] >= user_last_task_end[uid] and core_end_times[
                                   core_id] + task.exeTime <= task.deadLine]
            if candidate_cores:
                candidate_cores.sort(key=lambda x: (x[0], x[1]))  # 按时间和任务数量排序
                min_end_time, _, min_core_id = candidate_cores[0]
                task.startTime = core_end_times[min_core_id]
                cores[min_core_id].append(task)
                core_end_times[min_core_id] += task.exeTime
                user_last_task_end[uid] = core_end_times[min_core_id]
                inserted = True

            # 如果未找到匹配，查找由近到远的所有任务匹配
            if not inserted:
                candidate_cores = [(core_end_times[core_id], len(cores[core_id]), core_id) for core_id in range(m) if
                                   core_end_times[core_id] + task.exeTime <= task.deadLine]
                if candidate_cores:
                    candidate_cores.sort(key=lambda x: (x[0], x[1]))  # 按时间和任务数量排序
                    min_end_time, _, min_core_id = candidate_cores[0]
                    task.startTime = core_end_times[min_core_id]
                    cores[min_core_id].append(task)
                    core_end_times[min_core_id] += task.exeTime
                    user_last_task_end[uid] = core_end_times[min_core_id]
                    inserted = True

            # 如果仍未找到匹配，插入到用时最短的核中
            if not inserted:
                min_core_id = core_end_times.index(min(core_end_times))
                task.startTime = max(core_end_times[min_core_id], user_last_task_end[uid])
                cores[min_core_id].append(task)
                core_end_times[min_core_id] = task.startTime + task.exeTime
                user_last_task_end[uid] = core_end_times[min_core_id]

    # 4. 输出结果
    output_lines = []
    for coreId, core_tasks in enumerate(cores):
        line = str(len(core_tasks))
        for task in core_tasks:
            line += f" {task.msgType} {task.usrInst}"
        output_lines.append(line + "\n")

    print(''.join(output_lines), end='')


if __name__ == "__main__":
    main()
