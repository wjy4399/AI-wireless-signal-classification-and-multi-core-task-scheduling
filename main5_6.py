#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/5/10 下午8:33
# @File    : main.py
# @desc    : 智联杯示例代码
# @license : Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

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
ACCURACY = 0.69  # 假设模型预测准确率为69%

def adjust_execution_time(exeTime, accuracy):
    adjusted_time = exeTime * (2 - accuracy * accuracy)
    return round(adjusted_time)

def calculate_affinity_score(core_tasks):
    score = 0
    for i in range(1, len(core_tasks)):
        if core_tasks[i].msgType == core_tasks[i-1].msgType:
            score += 1
    return score

def calculate_total_score(affinity_score, overdue_tasks, total_tasks):
    capability_score = total_tasks - overdue_tasks
    total_score = affinity_score + capability_score
    absolute_upper_bound = 2 * total_tasks
    normalized_score = 100000 * math.log10(total_tasks) * (1 + (total_score - absolute_upper_bound) / absolute_upper_bound)
    return normalized_score

def main():
    # 1. 读取任务数、核数、系统最大执行时间
    n, m, c = map(int, input().split())

    # 2. 读取每个任务的信息
    tasks = [[] for _ in range(MAX_USER_ID)]
    all_tasks = []
    for idx in range(n):
        msgType, usrInst, exeTime, deadLine = map(int, input().split())
        exeTime = adjust_execution_time(exeTime, ACCURACY)  # 调整执行时间
        deadLine = min(deadLine, c)
        task = MessageTask(msgType, usrInst, exeTime, deadLine, idx)
        tasks[usrInst].append(task)
        all_tasks.append(task)

    # 3. 任务排序，优先级：最晚完成时间，执行时间
    for uid in range(MAX_USER_ID):
        tasks[uid].sort(key=lambda x: (x.deadLine, x.exeTime))

    # 4. 调度逻辑：贪心算法为每个用户实例找到最佳核，将用户实例的任务顺序不变地分配到该核上
    cores = [[] for _ in range(m)]
    core_end_times = [0] * m
    overdue_tasks = []

    for task in all_tasks:
        msgType, usrInst, exeTime, deadLine = task.msgType, task.usrInst, task.exeTime, task.deadLine
        assigned = False

        for core in range(m):
            if core_end_times[core] + exeTime <= deadLine:
                cores[core].append(task)
                core_end_times[core] += exeTime
                assigned = True
                break

        if not assigned:
            overdue_tasks.append(task)

    # 将超时的任务插入到最短用时的核心中
    for task in overdue_tasks:
        min_core = min(range(m), key=lambda x: sum(t.exeTime for t in cores[x]))
        cores[min_core].append(task)


    # 输出结果
    output_lines = []
    for coreId, core_tasks in enumerate(cores):
        core_tasks.sort(key=lambda x: x.idx)  # 按输入顺序排序
        line = f"{len(core_tasks)}"
        for task in core_tasks:
            line += f" {task.msgType} {task.usrInst}"
        output_lines.append(line)

    print('\n'.join(output_lines), end='')

if __name__ == "__main__":
    main()
