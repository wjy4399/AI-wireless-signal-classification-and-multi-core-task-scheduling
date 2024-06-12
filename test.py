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


MAX_USER_ID = 10005


def main():
    # 1. 读取任务数、核数、系统最大执行时间
    n, m, c = map(int, input().split())

    # 2. 读取每个任务的信息
    tasks = [[] for _ in range(MAX_USER_ID)]
    for _ in range(n):
        msgType, usrInst, exeTime, deadLine = map(int, input().split())
        deadLine = min(deadLine, c)
        task = MessageTask()
        task.msgType, task.usrInst, task.exeTime, task.deadLine = msgType, usrInst, exeTime, deadLine
        tasks[usrInst].append(task)

    # 3. 按照用户第一个任务的截止时间对用户进行排序
    user_tasks = [tasks[uid] for uid in range(MAX_USER_ID) if tasks[uid]]
    user_tasks.sort(key=lambda t: t[0].deadLine)

    # 4. 初始化每个核的任务队列和最终完成时间
    cores = [[] for _ in range(m)]
    core_endtime = [0] * m
    core_task_counts = [0] * m

    # 5. 逐个用户处理任务
    for user_task in user_tasks:
        user_task.sort(key=lambda t: t.deadLine)  # 对每个用户的任务按截止时间排序

        # 获取当前用户所有任务的类型集合
        user_msg_types = set(task.msgType for task in user_task)

        # 选择合适的核
        best_core = None
        best_match_count = -1
        min_endtime = float('inf')
        min_task_count = float('inf')

        for i in range(m):
            # 计算当前核中与用户所有任务类型相同的数量
            current_match_count = sum(1 for t in cores[i] if t.msgType in user_msg_types)
            if current_match_count > best_match_count:
                best_match_count = current_match_count
                best_core = i
            elif current_match_count == best_match_count and core_endtime[i] < min_endtime:
                min_endtime = core_endtime[i]
                best_core = i

        if best_core is None or (core_task_counts[best_core] - min(core_task_counts)) > min(core_task_counts) and (core_task_counts[best_core] - min(core_task_counts)) > 50:
            best_core = core_task_counts.index(min(core_task_counts))

        # 分配任务
        for task_index, task in enumerate(user_task):
            latest_start_time = task.deadLine - task.exeTime
            if core_endtime[best_core] <= latest_start_time:
                cores[best_core].append(task)
                core_endtime[best_core] += task.exeTime
            else:
                found_spot = False
                # 限制匹配范围在该用户的第n-1个任务
                user_prev_task_indices = [i for i, t in enumerate(cores[best_core]) if t.usrInst == task.usrInst and i < task_index]

                if user_prev_task_indices:
                    last_user_task_index = user_prev_task_indices[-1]
                else:
                    last_user_task_index = 0

                # 从核的最后一个任务向前匹配
                for j in range(len(cores[best_core]) - 1, last_user_task_index - 1, -1):
                    if cores[best_core][j].deadLine - cores[best_core][j].exeTime <= latest_start_time:
                        # 从该任务往前找相同类型任务
                        for k in range(j, last_user_task_index - 1, -1):
                            if cores[best_core][k].msgType == task.msgType:
                                cores[best_core].insert(k + 1, task)
                                found_spot = True
                                break
                        if not found_spot:
                            # 没有找到相同类型任务，则将任务放在该位置前面
                            cores[best_core].insert(j + 1, task)
                            found_spot = True
                        break

                if not found_spot:
                    # 如果在核的任务中没有找到合适位置，则将任务插入到末尾
                    cores[best_core].append(task)
                    core_endtime[best_core] += task.exeTime

            # 更新核的任务计数和结束时间
            core_task_counts[best_core] += 1
            core_endtime[best_core] = max(core_endtime[best_core], task.deadLine)

    # 6. 输出结果
    output_lines = []
    for coreId, core_tasks in enumerate(cores):
        line = str(len(core_tasks))
        for task in core_tasks:
            line += f" {task.msgType} {task.usrInst}"
        output_lines.append(line + "\n")

    print(''.join(output_lines), end='')


if __name__ == "__main__":
    main()
