#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/5/10 下午8:33
# @File    : main.py
# @desc    : 智联杯示例代码
# @license : Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import heapq
import math


class MessageTask:
    def __init__(self, msgType, usrInst, exeTime, deadLine, idx):
        self.msgType = msgType
        self.usrInst = usrInst
        self.exeTime = exeTime
        self.deadLine = deadLine
        self.idx = idx
        self.endTime = 0  # 新增属性记录任务的结束时间


MAX_USER_ID = 10005
ACCURACY = 0.69  # 假设模型预测准确率为65%


def adjust_execution_time(exeTime, accuracy):
    adjusted_time = exeTime * (2 - accuracy * accuracy)
    return round(adjusted_time)


def read_input_data():
    # 读取任务数、核数、系统最大执行时间
    n, m, c = map(int, input().split())

    # 读取每个任务的信息
    tasks = []
    for idx in range(n):
        msgType, usrInst, exeTime, deadLine = map(int, input().split())
        exeTime = adjust_execution_time(exeTime, ACCURACY)  # 调整执行时间
        deadLine = min(deadLine, c)
        tasks.append(MessageTask(msgType, usrInst, exeTime, deadLine, idx))

    return n, m, c, tasks


def schedule_tasks(n, m, tasks):
    # 对任务按截止时间和任务类型进行排序
    tasks.sort(key=lambda x: (x.deadLine, x.msgType))

    # 优先级队列管理核的可用时间
    pq = [(0, i) for i in range(m)]  # (time, core_id)
    heapq.heapify(pq)

    # 核分配结果
    cores = [[] for _ in range(m)]
    user_last_end_time = [0] * MAX_USER_ID  # 记录每个用户实例的最后完成时间

    for task in tasks:
        available_time, core_id = heapq.heappop(pq)

        # 确保同一用户的任务按顺序执行
        if user_last_end_time[task.usrInst] > available_time:
            available_time = user_last_end_time[task.usrInst]

        # 分配任务到选定的核
        cores[core_id].append(task)
        task.endTime = available_time + task.exeTime
        user_last_end_time[task.usrInst] = task.endTime

        # 更新核的可用时间
        heapq.heappush(pq, (task.endTime, core_id))

    return cores


def format_output(cores):
    output = []
    for core in cores:
        core_output = [len(core)]
        for task in core:
            core_output.extend([task.msgType, task.usrInst])
        output.append(core_output)
    return output


def calculate_scores(n, m, c, tasks, cores):
    affinity_score = 0
    completed_tasks = 0
    current_time = [0] * m

    task_dict = {(task.msgType, task.usrInst): (task.exeTime, task.deadLine) for task in tasks}

    for core in cores:
        if not core:
            continue

        for i in range(1, len(core)):
            if core[i].msgType == core[i - 1].msgType:
                affinity_score += 1

    for core_id, core in enumerate(cores):
        for task in core:
            exe_time, deadline = task_dict[(task.msgType, task.usrInst)]

            current_time[core_id] += exe_time
            if current_time[core_id] <= deadline:
                completed_tasks += 1

    total_score = affinity_score + completed_tasks
    absolute_upper_bound = 2 * n
    normalized_score = 100000 * math.log10(n) * (1 + (total_score - absolute_upper_bound) / absolute_upper_bound)

    return normalized_score, affinity_score, completed_tasks


def main():
    n, m, c, tasks = read_input_data()
    cores = schedule_tasks(n, m, tasks)
    output = format_output(cores)

    # 输出结果
    for core in output:
        print(' '.join(map(str, core)))



if __name__ == "__main__":
    main()
