# 计算并打印得分
score, affinity_score, completed_tasks = calculate_scores(n, m, c, tasks, cores)
print(f"Final Score: {score}")
print(f"Affinity Score: {affinity_score}")
print(f"Completed Tasks: {completed_tasks}")
for core_id, core_tasks in enumerate(cores):
    for task in core_tasks:
        print(f"Task({task.msgType}, {task.usrInst}) on Core {core_id} finishes at time {task.endTime}")

