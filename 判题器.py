import math
import os

def read_input_data(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    n, m, c = map(int, lines[0].split())
    tasks = []
    for line in lines[1:]:
        tasks.append(list(map(int, line.split())))

    return n, m, c, tasks

def read_output_data(output_file):
    with open(output_file, 'r') as f:
        lines = f.readlines()

    cores = []
    for line in lines:
        core_data = list(map(int, line.split()))
        num_tasks = core_data[0]
        tasks = [(core_data[i], core_data[i + 1]) for i in range(1, 2 * num_tasks + 1, 2)]
        cores.append(tasks)

    return cores

def adjust_execution_time(tasks, accuracy):
    adjusted_tasks = []
    adjustment_factor = 2 - accuracy ** 2
    for task in tasks:
        msg_type, usr_inst, exe_time, deadline = task
        adjusted_exe_time = int(exe_time * adjustment_factor)
        adjusted_tasks.append([msg_type, usr_inst, adjusted_exe_time, deadline])
    return adjusted_tasks

def check_constraints(n, m, c, tasks, cores):
    task_dict = {(task[0], task[1]): (task[2], task[3]) for task in tasks}
    user_task_completion_time = {}
    user_core_assignment = {}
    current_time = [0] * m
    incomplete_tasks = []

    for core_id, core in enumerate(cores):
        for task_idx, task in enumerate(core):
            if task not in task_dict:
                return False, f"Task {task} not found in the input data."

            exe_time, deadline = task_dict[task]
            msg_type, usr_inst = task

            # Ensure tasks of the same user run on the same core
            if usr_inst in user_core_assignment:
                if user_core_assignment[usr_inst] != core_id:
                    return False, f"Tasks for user instance {usr_inst} are not on the same core."
            else:
                user_core_assignment[usr_inst] = core_id

            # Ensure tasks are executed in sequence (FIFO) within a core
            if task_idx > 0 and core[task_idx - 1][1] == usr_inst and task_idx != core.index(task):
                return False, f"Tasks for user instance {usr_inst} in core {core_id} are not in FIFO order."

            current_time[core_id] += exe_time

            if usr_inst not in user_task_completion_time:
                user_task_completion_time[usr_inst] = []

            user_task_completion_time[usr_inst].append((current_time[core_id], deadline))

            if current_time[core_id] > deadline:
                incomplete_tasks.append((msg_type, usr_inst, exe_time, deadline, current_time[core_id]))

    # Check that tasks for the same user instance are in correct order across all cores
    for usr_inst, times in user_task_completion_time.items():
        times.sort(key=lambda x: x[1])  # Sort by deadline
        if times != sorted(times, key=lambda x: x[0]):
            return False, f"Tasks for user instance {usr_inst} are not completed in correct order across cores."

    # Ensure all tasks are assigned
    total_tasks_assigned = sum(len(core) for core in cores)
    if total_tasks_assigned != n:
        return False, f"Not all tasks are assigned. Expected {n}, but got {total_tasks_assigned}."

    if incomplete_tasks:
        print("Incomplete Tasks:")
        for task in incomplete_tasks:
            print(f"msg_type: {task[0]}, usr_inst: {task[1]}, exe_time: {task[2]}, deadline: {task[3]}, actual_completion_time: {task[4]}")
        return True, "There are incomplete tasks."

    return True, "All constraints satisfied."

def calculate_scores(n, m, c, tasks, cores):
    affinity_score = 0
    completed_tasks = 0
    current_time = [0] * m

    task_dict = {(task[0], task[1]): (task[2], task[3]) for task in tasks}

    for core in cores:
        if not core:
            continue

        for i in range(1, len(core)):
            if core[i][0] == core[i - 1][0]:
                affinity_score += 1

    for core_id, core in enumerate(cores):
        for task in core:
            exe_time, deadline = task_dict[task]

            current_time[core_id] += exe_time
            if current_time[core_id] <= deadline:
                completed_tasks += 1

    total_score = affinity_score + completed_tasks
    absolute_upper_bound = 2 * n
    normalized_score = 100000 * math.log10(n) * (1 + (total_score - absolute_upper_bound) / absolute_upper_bound)

    return normalized_score, affinity_score, completed_tasks

# Example usage:
input_file = 'dataset/多核任务调度数据集/case2.txt'
output_files = f'result-case2'
accuracy = 0.8
for root, dirs, files in os.walk(output_files):
        # 遍历文件
        for file in files:
            # 检查文件是否是txt文件
            if file.endswith(".txt"):
                output_name=os.path.splitext(file)[0]

                output_file = f'{output_files}/{file}'  # Replace with your output file
                n, m, c, tasks = read_input_data(input_file)
                tasks = adjust_execution_time(tasks, accuracy)
                cores = read_output_data(output_file)

                constraints_satisfied, message = check_constraints(n, m, c, tasks, cores)
                print(f'__________________{output_name}__________________')
                if constraints_satisfied:
                    score, affinity_score, completed_tasks = calculate_scores(n, m, c, tasks, cores)
                    print(f"Final Score: {score}")
                    print(f"Affinity Score: {affinity_score}")
                    print(f"Completed Tasks: {completed_tasks}")
                else:
                    print(f"Final Score: 0 (Constraints not satisfied: {message})")
