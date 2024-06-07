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
ACCURACY = 0.65  # 假设模型预测准确率为69%


def adjust_execution_time(exeTime, accuracy):
    adjusted_time = exeTime * (2 - accuracy * accuracy)
    return round(adjusted_time)


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

    # 4. 调度逻辑：为每个用户实例找到最佳核，将用户实例的任务顺序不变地分配到该核上
    cores = [[] for _ in range(m)]
    core_end_times = [0] * m
    affinity_scores = [0] * m

    for uid in range(MAX_USER_ID):
        if not tasks[uid]:
            continue

        task_total_time = sum(task.exeTime for task in tasks[uid])
        min_deadline_time = min(task.deadLine for task in tasks[uid])

        # 找到在不超时情况下亲和力最高的核
        best_core = -1
        max_affinity = -1
        min_end_time = float('inf')
        for i in range(m):
            current_affinity = 0
            if cores[i] and cores[i][-1].msgType == tasks[uid][0].msgType:
                current_affinity = 1

            if core_end_times[i] + task_total_time <= min_deadline_time:
                if current_affinity > max_affinity or (
                        current_affinity == max_affinity and core_end_times[i] < min_end_time):
                    max_affinity = current_affinity
                    min_end_time = core_end_times[i]
                    best_core = i

        # 如果没有找到满足条件的核，选择当前负载最小的核
        if best_core == -1:
            best_core = core_end_times.index(min(core_end_times))

        last_msg_type = None
        for task in tasks[uid]:
            if cores[best_core] and cores[best_core][-1].msgType == task.msgType:
                affinity_scores[best_core] += 1
            task.endTime = core_end_times[best_core] + task.exeTime  # 记录任务的结束时间
            cores[best_core].append(task)
            core_end_times[best_core] += task.exeTime
            last_msg_type = task.msgType

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
