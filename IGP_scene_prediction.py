import numpy as np
import GPy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def gp_flow_regress(time, dir, length, var):

    k_mat = GPy.kern.Matern52(input_dim=1, variance=var, lengthscale=length)
    k_lin = GPy.kern.Linear(1)
    k = k_mat
    mod = GPy.models.GPRegression(time, dir, k)
    mod.optimize(messages=False)
    mod.optimize_restarts(num_restarts=10)

    return mod


def agent_regress(traj, length, var):
    """
    Regress each agent with a Gaussian process
    :param traj: 2xT array of xy-coordinate trajectory of agent to interpret into the GP posterior
    :return: a GP model of an agent trajectory
    """

    #TODO: regress x- y- coordinate saparately according to he time points

    time = traj[:, 0].reshape(len(traj[:, 0]), 1)
    x_ro = traj[0, 1]
    x_dir = traj[:, 1] - x_ro
    x_dir = x_dir.reshape(len(x_dir), 1)
    mod_x = gp_flow_regress(time, x_dir, length, var)

    y_ro = traj[0, 2]
    y_dir = traj[:, 2] - y_ro
    y_dir = y_dir.reshape(len(y_dir), 1)
    mod_y = gp_flow_regress(time, y_dir, length, var)

    m_xy = [mod_x, mod_y, x_ro, y_ro]

    return m_xy


def interact_potential(coord, h, alpha):
    """
    output the interaction potential of the sample
    :param samp_traj: trajectory sample from each agents' GP model
    :param h: parameter h
    :param alpha: parameter alpha
    :return: the interaction potential of the sample
    """
    result = 1
    for i in range(0,len(coord)-1):
        traj_i = coord[i, :, :]
        if np.all(traj_i == 0):
            continue
        for j in range(i+1, len(coord)):
            traj_j = coord[j, :, :]
            if np.all(traj_j == 0):
                continue
            for k in range(len(traj_j[0])):
                temp = 1-alpha*np.exp((-1/h**2)*np.linalg.norm(traj_i[:, k]-traj_j[:, k]))
                result = result*temp
    return result


def path_sample(mods, time_axis):
    sample_path = np.zeros((len(mods), len(time_axis), 2))
    for i in range(len(mods)):
        agt_mod = mods[i]
        if agt_mod is None:
            continue
        x_mod = agt_mod[0]
        y_mod = agt_mod[1]
        samp_x = x_mod.posterior_samples_f(time_axis, 1).reshape((-1, 1)) + agt_mod[2]
        samp_y = y_mod.posterior_samples_f(time_axis, 1).reshape((-1, 1)) + agt_mod[3]
        # samp_x = x_mod.sample_y(time_axis, 1).reshape((-1, 1))
        # samp_y = y_mod.sample_y(time_axis, 1).reshape((-1, 1))
        coord = np.column_stack((samp_x, samp_y))
        sample_path[i, :, :] = coord
    return sample_path


def path_prediction(mods, t_points, samp_num, h_value):
    """
    find the predicted path of the overall crowds with the max interaction potential
    :param mods: GP model of each agent and provisional ROBOT
    :param t_points: time points desired to sample
    :param samp_num: number of samples want to look at
    :return: trajectory with the max interaction potential at the time span
    """
    h = h_value  # change the parameter h = 24.76
    alpha = 0.99
    time_points = t_points.reshape((-1,1))
    sample_collection = np.zeros((samp_num, len(mods), len(time_points), 2))
    interact_pot = np.empty((samp_num, 1))
    for i in range(0, samp_num):
        sample_path = path_sample(mods, time_points)
        sample_collection[i, :, :, :] = sample_path
        interact_pot[i, :] = interact_potential(sample_path, h, alpha)
    m_index = np.argmax(interact_pot)
    max_path = sample_collection[m_index, :, :, :]

    #TODO: sample the trajectory for the crowds
    # calculate the interaction potential
    # find the maximum one

    return max_path


def time_match(agt, t_point):
    curr_traj = agt[agt[:, 0] < t_point]
    return curr_traj


def navigation(obsv, t_span, l_scale, var_scale, s_amount):
    """
    navigate the provisional ROBOT through the crowd
    :param obsv: observed trajectory of the crowd
    :param t_span: time span that the ROBOT walk
    :param destination: beginning and destination coordinate of the provisional ROBOT，with a time column
    :param l_scale: length scale for the GP_regression
    :param var_scale: variance hyperparameter for the GP_regression
    :param s_amount: amount of sample draw for IGP
    :return: trajectory the provisional ROBOT takes
    """

    navi_result = obsv
    curr_span = np.array([])
    for t_point in t_span:
        print(t_point)
        mods = [None]*len(navi_result)
        curr_span = np.append(curr_span, t_point)
        counter = 0
        for i in range(0, len(navi_result)):
            agt = navi_result[i]
            print(counter)
            counter += 1
            agt_traj = time_match(agt, t_point)
            if len(agt_traj) == 0:
                continue
            agt_mod = agent_regress(agt_traj, l_scale, var_scale)
            mods[i] = agt_mod
        min_distance_collect = []
        for i in range(0, len(navi_result)-1):
            agt = navi_result[i]
            agt_i_pos = agt[agt[:, 0] < t_point]
            if len(agt_i_pos) == 0:
                continue
            for j in range(i+1, len(navi_result)):
                agt = navi_result[j]
                agt_j_pos = agt[agt[:, 0] < t_point]
                if len(agt_j_pos) == 0:
                    continue
                c = (agt_i_pos[-1, 1] - agt_j_pos[-1, 1]) ** 2 + (agt_i_pos[-1, 2] - agt_j_pos[-1, 2]) ** 2
                dist = math.sqrt(c)
                min_distance_collect.append(dist)
        numpy_min_collect = np.array(min_distance_collect)
        h_value = numpy_min_collect.min()
        optimized_path = path_prediction(mods, curr_span, s_amount, h_value)  # change the number sample
        for i in range(0, len(navi_result)):
            agt = navi_result[i]
            agt_traj = time_match(agt, t_point)
            if len(agt_traj) < 8:
                continue
            opt_path_i = np.column_stack((curr_span, optimized_path[i, :, :]))
            next_location = opt_path_i[-1, :]
            agt = np.vstack((agt, next_location))
            navi_result[i] = agt

    return navi_result, mods

def data_clean(start_point, end_point, interval_len):
    mat = np.genfromtxt("data/annotations.txt")  # read file
    mat_str = np.genfromtxt("data/annotations.txt", dtype='str')
    agent_pool = mat[mat[:, 6] == 0]

    agents_present = agent_pool[np.logical_and(agent_pool[:, 5] < end_point+1, agent_pool[:, 5] > start_point-1)]
    agents_number = np.unique(agents_present[:, 0])

    agent = []
    for i in agents_number:
        temp = agents_present[agents_present[:, 0] == i]
        temp = temp[temp[:, 5] % interval_len == 0]
        x_coord = (temp[:, 3] - temp[:, 1]) / 2 + temp[:, 1]
        y_coord = (temp[:, 4] - temp[:, 2]) / 2 + temp[:, 2]
        agent_temp = np.column_stack((temp[:, 5], x_coord, y_coord))
        agent_temp[:, 0] = agent_temp[:, 0] - 300
        agent.append(agent_temp)

    observation = []
    for i in agent:
        if len(i) < 8:
            observation.append(i)
            continue
        observation.append(i[0:8, :])

    time_span = np.array([range(interval_len*8, interval_len*20+1, 12)]).reshape((-1, 1))

    return agent, observation, time_span

def overall_plot(agent, result):
    img = mpimg.imread('data/reference.jpg')
    trej_ori = agent
    trej_result = result
    for i in range(0, len(agent)):
        plt.imshow(img)
        plt.plot(trej_ori[i][:, 1], trej_ori[i][:, 2], "green")
        plt.plot(trej_result[i][0:len(trej_ori[i]), 1], trej_result[i][0:len(trej_ori[i]), 2], "red")
        plt.plot(trej_ori[i][0:8, 1], trej_ori[i][0:8, 2], "blue")
        plt.axis([0, 1409, 1916, 0])
    plt.show()
