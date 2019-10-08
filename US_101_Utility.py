import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def data_clean(start_point, end_point, interval_len):
    mat = np.genfromtxt("data/us101lane1.csv", delimiter=",")# read file
    agents_present = mat[np.logical_and(mat[:, 1] < end_point + 1, mat[:, 1] > start_point - 1)]
    agents_number = np.unique(agents_present[:, 0])
    agent = []
    for i in agents_number:
        temp = agents_present[agents_present[:, 0] == i]
        temp = temp[temp[:, 6] == temp[0, 6]]
        temp = temp[temp[:, 7] == temp[0, 7]]
        temp = temp[temp[:, 1] % interval_len == (start_point % interval_len)]
        agent_temp = np.column_stack((temp[:, 1], temp[:, 2], temp[:, 3]))
        agent_temp[:, 0] = agent_temp[:, 0] - start_point
        agent.append(agent_temp)

    observation = []
    for i in agent:
        if len(i) < 12:
            observation.append(i)
            continue
        observation.append(i[0:12, :])

    time_span = np.array([range(interval_len*12, interval_len*36+1, interval_len)]).reshape((-1, 1))

    return agent, observation, time_span, agents_number


def ade_and_fde(agent, current_interval):
    ade = np.zeros((len(agent) + 1, 1))
    fde = np.zeros((len(agent) + 1, 1))
    legal_agent = len(agent)
    for j in range(0, len(agent)):
        ade_sum_single = 0
        g_truth = agent[j]
        navi = current_interval[j]
        if len(g_truth) <= 12:
            legal_agent -= 1
            continue
        for k in range(12, len(g_truth)):
            print(j, k)
            d_error = (g_truth[k, 1] - navi[k, 1]) ** 2 + (g_truth[k, 2] - navi[k, 2]) ** 2
            d_error = math.sqrt(d_error)
            ade_sum_single += d_error
        ade[j, 0] = ade_sum_single / len(g_truth)
        fd_error = (g_truth[-1, 1] - navi[-1, 1]) ** 2 + (g_truth[-1, 2] - navi[-1, 2]) ** 2
        fde[j, 0] = math.sqrt(fd_error)
    ade[-1, 0] = np.sum(ade[0:len(agent), 0]) / legal_agent
    fde[-1, 0] = np.sum(fde[0:len(agent), 0]) / legal_agent

    return ade, fde