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
    tasks = [[] for _ in range(MAX_USER_ID)]
    for _ in range(n):
        msgType, usrInst, exeTime, deadLine = map(int, input().split())
        task = MessageTask()
        task.msgType, task.usrInst, task.exeTime, task.deadLine = msgType, usrInst, exeTime, deadLine
        tasks[usrInst].append(task)

    # 按任务截止时间排序
    for user_tasks in tasks:
        user_tasks.sort(key=lambda x: x.deadLine)

    # 初始化核状态
    cores = [[] for _ in range(m)]
    core_end_times = [0] * m  # 记录每个核的结束时间
    user_core_map = {}  # 记录每个用户分配到的核

    # 3. 调度逻辑：按截止时间排序并调度
    for uid in range(MAX_USER_ID):
        if not tasks[uid]:
            continue

        for task in tasks[uid]:
            if uid not in user_core_map:
                # 如果是该用户的第一个任务
                candidate_cores = []
                for core_id in range(m):
                    same_type_sequences = []
                    current_sequence = []
                    for t in cores[core_id]:
                        if t.msgType == task.msgType:
                            current_sequence.append(t)
                        else:
                            if current_sequence:
                                same_type_sequences.append((len(current_sequence), core_id, current_sequence[-1]))
                                current_sequence = []
                    if current_sequence:
                        same_type_sequences.append((len(current_sequence), core_id, current_sequence[-1]))

                    if same_type_sequences:
                        same_type_sequences.sort(reverse=True, key=lambda x: (x[0], x[2].startTime))
                        candidate_cores.append(same_type_sequences[0])

                if candidate_cores:
                    _, min_core_id, last_task = max(candidate_cores, key=lambda x: (x[0], x[2].startTime))
                    task.startTime = last_task.startTime + last_task.exeTime
                    cores[min_core_id].append(task)
                    core_end_times[min_core_id] = task.startTime + task.exeTime
                    user_core_map[uid] = min_core_id
                else:
                    # 如果没有相同类型的任务
                    min_core_id = core_end_times.index(min(core_end_times))
                    task.startTime = core_end_times[min_core_id]
                    cores[min_core_id].append(task)
                    core_end_times[min_core_id] += task.exeTime
                    user_core_map[uid] = min_core_id
            else:
                # 如果不是该用户的第一个任务，则分配到同一个核中
                core_id = user_core_map[uid]
                last_task_end_time = max([t.startTime + t.exeTime for t in cores[core_id] if t.usrInst == uid])
                same_type_sequences = []
                current_sequence = []
                for t in cores[core_id]:
                    if t.startTime >= last_task_end_time:
                        if t.msgType == task.msgType:
                            current_sequence.append(t)
                        else:
                            if current_sequence:
                                same_type_sequences.append((len(current_sequence), current_sequence[-1]))
                                current_sequence = []
                if current_sequence:
                    same_type_sequences.append((len(current_sequence), current_sequence[-1]))

                if same_type_sequences:
                    same_type_sequences.sort(reverse=True, key=lambda x: x[0])
                    _, last_task = same_type_sequences[0]
                    task.startTime = last_task.startTime + last_task.exeTime
                else:
                    task.startTime = core_end_times[core_id]

                cores[core_id].append(task)
                core_end_times[core_id] = task.startTime + task.exeTime

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
