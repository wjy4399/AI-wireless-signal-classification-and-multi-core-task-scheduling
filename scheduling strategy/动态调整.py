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
        self.reserveTime =0
        self.latest_start_time=0


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

    force_least_task_core = False  # 是否强制分配到最少任务的核

    # 5. 逐个用户处理任务
    for user_task in user_tasks:
        user_task.sort(key=lambda t: t.deadLine)  # 对每个用户的任务按截止时间排序

        if force_least_task_core:
            best_core = core_task_counts.index(min(core_task_counts))
        else:
            # 获取当前用户所有任务的类型集合
            user_msg_types = set(task.msgType for task in user_task)

            # 选择合适的核
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
                    min_endtime = core_endtime[i]  # 更新最小结束时间
                    min_task_count = core_task_counts[i]  # 更新最小任务数
                    best_core = i
                elif current_match_count == best_match_count:
                    if core_endtime[i] < min_endtime or (
                            core_endtime[i] == min_endtime and core_task_counts[i] < min_task_count):
                        min_endtime = core_endtime[i]
                        min_task_count = core_task_counts[i]
                        best_core = i

            # 处理没有找到合适的核或当前最优核的任务数目大于最小任务数目50以上的情况
            if best_core is None or (core_task_counts[best_core] - min(core_task_counts)) > 50:
                best_core = core_task_counts.index(min(core_task_counts))

        # 分配任务
        for task_index, task in enumerate(user_task):
            task.latest_start_time = task.deadLine - task.exeTime
            task.reserveTime = task.deadLine
            found_spot = False
            if core_endtime[best_core] == 0:
                cores[best_core].insert(0, task)
                task.startTime = 0
                core_endtime[best_core] += task.exeTime
                task.reserveTime = task.deadLine - task.exeTime
                print(f" 用户的第{task_index}个任务 {task.msgType} {task.usrInst} 插入核{best_core}第0位  核任务为0")

            else:
                if task_index == 0:  # 处理该用户的第一个任务
                    if core_endtime[best_core] <= task.latest_start_time:
                        for j in range(len(cores[best_core]) - 1, -1, -1):
                            if cores[best_core][j].reserveTime > task.exeTime:
                                cores[best_core][j].startTime = cores[best_core][j].startTime + task.exeTime
                                cores[best_core][j].reserveTime = cores[best_core][j].reserveTime - task.exeTime
                                consecutive_count = 0
                                if cores[best_core][j].msgType == task.msgType:
                                        consecutive_count += 1
                                        cores[best_core].insert(j + 1, task)
                                        print(f" 用户的第{task_index}个任务 {task.msgType} {task.usrInst} 插入核{best_core}第{j + 1}位   {core_endtime[best_core]}<={task.latest_start_time} 匹配亲和")
                                        cores[best_core][j].startTime = cores[best_core][j].startTime - task.exeTime
                                        cores[best_core][j].reserveTime = cores[best_core][j].reserveTime + task.exeTime
                                        core_endtime[best_core] += task.exeTime
                                        task.startTime = cores[best_core][j].startTime + cores[best_core][j].exeTime
                                        found_spot = True
                                        break
                            else:
                                cores[best_core].insert(j + 1, task)
                                print(f" 用户的第{task_index}个任务 {task.msgType} {task.usrInst} 插入核{best_core}第{j + 1}位 {core_endtime[best_core]}<={task.latest_start_time} 第{j}位{cores[best_core][j].msgType} {cores[best_core][j].usrInst}富余时间不够")
                                core_endtime[best_core] += task.exeTime
                                task.startTime = cores[best_core][j].startTime + cores[best_core][j].exeTime
                                found_spot = True
                                break
                        if not found_spot:
                            cores[best_core].insert(0 , task)
                            print(f" 用户的第{task_index}个任务  {task.msgType} {task.usrInst} 插入核{best_core}第0位   {core_endtime[best_core]}<={task.latest_start_time}  不亲和， 其他任务富余时间够")
                            task.startTime = 0
                            task.reserveTime = task.deadLine - task.exeTime
                            core_endtime[best_core] += task.exeTime
                            continue
                    else:

                        for j in range(len(cores[best_core]) - 1, -1, -1):
                            if found_spot:
                                break
                            if cores[best_core][j].startTime <= task.latest_start_time:
                                if found_spot:
                                    break
                                for k in range(j, -1, -1):
                                    if found_spot:
                                        break
                                    if cores[best_core][k].reserveTime > task.exeTime:
                                        cores[best_core][k].startTime = cores[best_core][k].startTime + task.exeTime
                                        cores[best_core][k].reserveTime = cores[best_core][k].reserveTime - task.exeTime
                                        consecutive_count = 0
                                        if cores[best_core][k].msgType == task.msgType:
                                                consecutive_count += 1
                                                cores[best_core].insert(k + 1, task)
                                                print(f" 用户的第{task_index}个任务  {task.msgType} {task.usrInst} 插入核{best_core}第{k + 1}位  {core_endtime[best_core]}>{task.latest_start_time}  匹配亲和")
                                                cores[best_core][k].startTime = cores[best_core][
                                                                                    k].startTime - task.exeTime
                                                cores[best_core][k].reserveTime = cores[best_core][
                                                                                      k].reserveTime + task.exeTime
                                                core_endtime[best_core] += task.exeTime
                                                task.startTime = cores[best_core][k].startTime + cores[best_core][
                                                    k].exeTime
                                                found_spot = True
                                                break
                                    else:
                                        cores[best_core].insert(k + 1, task)
                                        print(f" 用户的第{task_index}个任务  {task.msgType} {task.usrInst} 插入核{best_core}第{k + 1}位 {core_endtime[best_core]}>{task.latest_start_time}  第{k}位{cores[best_core][k].msgType} {cores[best_core][k].usrInst}富余时间不够")
                                        core_endtime[best_core] += task.exeTime
                                        task.startTime = cores[best_core][k].startTime + cores[best_core][k].exeTime
                                        found_spot = True
                                        break
                                if not found_spot:
                                    cores[best_core].insert(0, task)
                                    print(f" 用户的第{task_index}个任务 {task.msgType} {task.usrInst} {core_endtime[best_core]}>{task.latest_start_time} 插入核{best_core}第0位")
                                    task.startTime = 0
                                    task.reserveTime = task.deadLine - task.exeTime
                                    core_endtime[best_core] += task.exeTime
                                    found_spot = True
                                    break
                else:  # 处理该用户的第n个任务，n不等于1
                    last_user_task_index = None
                    for j in range(len(cores[best_core]) - 1, -1, -1):
                        if cores[best_core][j].usrInst == task.usrInst:
                            last_user_task_index = j
                            break
                    if core_endtime[best_core]==cores[best_core][last_user_task_index].startTime+cores[best_core][last_user_task_index].exeTime:
                        cores[best_core].insert(last_user_task_index + 1, task)
                        print(f" 用户的第{task_index} n个任务 {task.msgType} {task.usrInst} 插入核{best_core}第{last_user_task_index + 1}位  最后一位等于上一个任务  {cores[best_core][last_user_task_index].msgType} {cores[best_core][last_user_task_index].usrInst} ")
                        task.startTime = core_endtime[best_core]
                        core_endtime[best_core] += task.exeTime
                        continue
                    if core_endtime[best_core] <= task.latest_start_time:
                        for j in range(len(cores[best_core]) - 1, last_user_task_index - 1, -1):
                            if cores[best_core][j].reserveTime > task.exeTime:
                                cores[best_core][j].startTime = cores[best_core][j].startTime + task.exeTime
                                cores[best_core][j].reserveTime = cores[best_core][j].reserveTime - task.exeTime
                                consecutive_count = 0
                                if cores[best_core][j].msgType == task.msgType:
                                        consecutive_count += 1
                                        cores[best_core].insert(j + 1, task)
                                        print(f" 用户的第{task_index} n 个任务  {task.msgType} {task.usrInst} 插入核{best_core}第{j + 1}位 亲和 {core_endtime[best_core]}<={task.latest_start_time} ")
                                        cores[best_core][j].startTime = cores[best_core][
                                                                            j].startTime - task.exeTime
                                        cores[best_core][j].reserveTime = cores[best_core][
                                                                              j].reserveTime + task.exeTime
                                        core_endtime[best_core] += task.exeTime
                                        task.startTime = cores[best_core][j].startTime + cores[best_core][j].exeTime
                                        found_spot = True
                                        break
                            else:
                                cores[best_core].insert(j + 1, task)
                                print(
                                    f" 用户的第{task_index} n个任务 {task.msgType} {task.usrInst} 插入核{best_core}第{j + 1}位 {core_endtime[best_core]}<={task.latest_start_time} 第{j}位{cores[best_core][j].msgType} {cores[best_core][j].usrInst}富余时间不够")
                                core_endtime[best_core] += task.exeTime
                                task.startTime = cores[best_core][j].startTime + cores[best_core][j].exeTime
                                found_spot = True
                                break

                        if not found_spot:
                            cores[best_core].insert(last_user_task_index+1 , task)
                            print(f" 用户的第{task_index} n个任务  {task.msgType} {task.usrInst} 插入核{best_core}第{last_user_task_index+1}位   {core_endtime[best_core]}<={task.latest_start_time}  不亲和 其他任务富余时间够  循环到该用户上个任务的下一位")
                            cores[best_core][last_user_task_index].startTime = cores[best_core][
                                                                last_user_task_index].startTime - task.exeTime
                            cores[best_core][last_user_task_index].reserveTime = cores[best_core][
                                                                  last_user_task_index].reserveTime + task.exeTime
                            core_endtime[best_core] += task.exeTime
                            task.startTime = cores[best_core][last_user_task_index].startTime + cores[best_core][last_user_task_index].exeTime
                            found_spot = True

                    else:
                        found_spot = False
                        for j in range(len(cores[best_core]) - 1, last_user_task_index - 1, -1):
                            if found_spot:
                                break
                            if cores[best_core][j].startTime <= task.latest_start_time:
                                for k in range(j, last_user_task_index - 1, -1):
                                    if found_spot:
                                        break
                                    if cores[best_core][k].reserveTime > task.exeTime:
                                        cores[best_core][k].startTime = cores[best_core][k].startTime + task.exeTime
                                        cores[best_core][k].reserveTime = cores[best_core][k].reserveTime - task.exeTime
                                        consecutive_count = 0
                                        if cores[best_core][k].msgType == task.msgType:
                                                consecutive_count += 1

                                                cores[best_core].insert(k + 1, task)
                                                print(
                                                    f" 用户的第{task_index} n个任务  {task.msgType} {task.usrInst} 插入核{best_core}第{k + 1}位  {core_endtime[best_core]}>{task.latest_start_time}  匹配亲和")
                                                cores[best_core][k].startTime = cores[best_core][
                                                                                    k].startTime - task.exeTime
                                                cores[best_core][k].reserveTime = cores[best_core][
                                                                                      k].reserveTime + task.exeTime
                                                core_endtime[best_core] += task.exeTime
                                                task.startTime = cores[best_core][k].startTime + cores[best_core][
                                                    k].exeTime
                                                found_spot = True
                                                break
                                    else:
                                        cores[best_core].insert(k + 1, task)
                                        print(f" 用户的第{task_index} n个任务  {task.msgType} {task.usrInst} 插入核{best_core}第{k + 1}位 {core_endtime[best_core]}>{task.latest_start_time}  第{k}位{cores[best_core][k].msgType} {cores[best_core][k].usrInst}富余时间不够")

                                        core_endtime[best_core] += task.exeTime
                                        task.startTime = cores[best_core][k].startTime + cores[best_core][k].exeTime
                                        found_spot = True
                                        break
                                if not found_spot:
                                    cores[best_core].insert(last_user_task_index+1, task)
                                    print(f" 用户的第{task_index} n个任务 {task.msgType} {task.usrInst} {core_endtime[best_core]}>{task.latest_start_time} 插入核{best_core}第{last_user_task_index+1}位，该用户上一个任务的后面")
                                    cores[best_core][k].startTime = cores[best_core][
                                                                        last_user_task_index].startTime - task.exeTime
                                    cores[best_core][k].reserveTime = cores[best_core][
                                                                          last_user_task_index].reserveTime + task.exeTime
                                    core_endtime[best_core] += task.exeTime
                                    task.startTime = cores[best_core][last_user_task_index].startTime + cores[best_core][last_user_task_index].exeTime
                                    found_spot = True
                                    break

                            # 更新核的任务计数和结束时间
            core_task_counts[best_core] += 1
        # 检查任务数量差异
        max_tasks = max(core_task_counts)
        min_tasks = min(core_task_counts)
        if max_tasks - min_tasks > 100:
            force_least_task_core = True
        elif max_tasks - min_tasks < 50:
            force_least_task_core = False

    # 6. 检查并纠正连续相同类型的任务
    for core in cores:
        i = 0
        while i < len(core):
            start = i
            while i < len(core) - 1 and core[i].msgType == core[i + 1].msgType:
                i += 1
            if i > start:
                same_type_tasks = core[start:i + 1]
                same_type_tasks.sort(key=lambda t: t.deadLine)
                core[start:i + 1] = same_type_tasks
            i += 1

    # 7. 输出结果
    output_lines = []
    for coreId, core_tasks in enumerate(cores):
        line = str(len(core_tasks))
        for task in core_tasks:
            line += f" {task.msgType} {task.usrInst}"
        output_lines.append(line + "\n")

    print(''.join(output_lines), end='')


if __name__ == "__main__":
    main()
