import numpy as np
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import tensorflow as tf
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pylab
import math


def gp_flow_regress(time, dir):

    k_mat = GPy.kern.Matern52(input_dim=1, variance=0.01, lengthscale=1)
    k_lin = GPy.kern.Linear(1)
    k = k_mat
    mod = GPy.models.GPRegression(time, dir, k)
    mod.optimize(messages=False)
    mod.optimize_restarts(num_restarts=10)

    # mod = GaussianProcessRegressor(kernel=Matern(length_scale=30, length_scale_bounds=(30, 65)),
    #                                normalize_y=True, n_restarts_optimizer=3)
    # mod.fit(time, dir)

    return mod


def agent_regress(traj):
    """
    Regress each agent with a Gaussian process
    :param traj: 2xT array of xy-coordinate trajectory of agent to interpret into the GP posterior
    :return: a GP model of an agent trajectory
    """

    #TODO: regress x- y- coordinate saparately according to he time points

    time = traj[:, 0].reshape(len(traj[:, 0]), 1)
    x_dir = traj[:, 1]
    x_dir = x_dir.reshape(len(x_dir), 1)
    mod_x = gp_flow_regress(time, x_dir)

    y_dir = traj[:, 2]
    y_dir = y_dir.reshape(len(y_dir), 1)
    mod_y = gp_flow_regress(time, y_dir)

    m_xy = [mod_x, mod_y]

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
        samp_x = x_mod.posterior_samples_f(time_axis, 1) .reshape((-1, 1))
        samp_y = y_mod.posterior_samples_f(time_axis, 1).reshape((-1, 1))
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


def navigation(obsv, t_span):
    """
    navigate the provisional ROBOT through the crowd
    :param obsv: observed trajectory of the crowd
    :param t_span: time span that the ROBOT walk
    :param destination: beginning and destination coordinate of the provisional ROBOT，with a time column
    :return: trajectory the provisional ROBOT takes
    """
    #TODO: for each timepoint
    # regress each agent at certain time span [use agent_regress() for each agent]
    # regress the provisional ROBOT with coordinate it walked [agent_regress()]
    # find the predicted path with maximum interaction_potential
    # put the provisional ROBOT's location in to its path
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
            agt_mod = agent_regress(agt_traj)
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
        optimized_path = path_prediction(mods, curr_span, 1000, h_value)  # change the number sample
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

if __name__ == '__main__':

    mat = np.genfromtxt("data/annotations.txt")  # read file
    mat_str = np.genfromtxt("data/annotations.txt", dtype='str')
    agent_pool = mat[mat[:, 6] == 0]

    result_collect = []
    mods_collect = []
    for i in range(0, 7):
        print("processing number", i)
        agents_present = agent_pool[np.logical_and(agent_pool[:, 5] < 300+28*(i+1),
                                                       agent_pool[:, 5] > 299+28*i)]
        agents_number = np.unique(agents_present[:, 0])
        agent = []
        for num in agents_number:
            temp = agents_present[agents_present[:, 0] == num]
            x_coord = (temp[:, 3] - temp[:, 1]) / 2 + temp[:, 1]
            y_coord = (temp[:, 4] - temp[:, 2]) / 2 + temp[:, 2]
            agent_temp = np.column_stack((temp[:, 5], x_coord, y_coord))
            agent.append(agent_temp)
        observation = []
        for agt in agent:
            if len(agt) < 8:
                observation.append(agt)
                continue
            observation.append(agt[0:8, :])
        time_span = np.array([range(28*i+300, 28*(i+1)+300)]).reshape((28, 1))
        test, mods = navigation(observation, time_span[8:28, :])
        result_collect.append(test)
        mods_collect.append(mods)

    img = mpimg.imread('data/reference.jpg')
    for i in range(0, 7):
        print("processing number", i)
        agents_present = agent_pool[np.logical_and(agent_pool[:, 5] < 300+28*(i+1),
                                                       agent_pool[:, 5] > 299+28*i)]
        agents_number = np.unique(agents_present[:, 0])
        agent = []
        for num in agents_number:
            temp = agents_present[agents_present[:, 0] == num]
            x_coord = (temp[:, 3] - temp[:, 1]) / 2 + temp[:, 1]
            y_coord = (temp[:, 4] - temp[:, 2]) / 2 + temp[:, 2]
            agent_temp = np.column_stack((temp[:, 5], x_coord, y_coord))
            agent.append(agent_temp)
        current_interval = result_collect[i]
        plt.imshow(img)
        for j in range(0, len(agent)):
            plt.plot(current_interval[j][:, 1], current_interval[j][:, 2], "blue")
            plt.plot(agent[j][:, 1], agent[j][:, 2], "red")
        plt.axis([0, 1409, 1916, 0])
        plt.show()

    ade_collect = []
    fde_collect = []
    for i in range(0, 7):
        print("processing number", i)
        agents_present = agent_pool[np.logical_and(agent_pool[:, 5] < 300+28*(i+1),
                                                       agent_pool[:, 5] > 299+28*i)]
        agents_number = np.unique(agents_present[:, 0])
        agent = []
        for num in agents_number:
            temp = agents_present[agents_present[:, 0] == num]
            x_coord = (temp[:, 3] - temp[:, 1]) / 2 + temp[:, 1]
            y_coord = (temp[:, 4] - temp[:, 2]) / 2 + temp[:, 2]
            agent_temp = np.column_stack((temp[:, 5], x_coord, y_coord))
            agent.append(agent_temp)
        current_interval = result_collect[i]
        ade = np.empty((len(agent)+1, 1))
        fde = np.empty((len(agent) + 1, 1))
        for j in range(0, len(agent)):
            ade_sum_single = 0
            g_truth = agent[j]
            navi = current_interval[j]
            for k in range(0, len(g_truth)):
                d_error = (g_truth[k, 1] - navi[k, 1])**2 + (g_truth[k, 2] - navi[k, 2])**2
                d_error = math.sqrt(d_error)
                ade_sum_single+=d_error
            ade[j, 0] = ade_sum_single/len(g_truth)
            fd_error = (g_truth[-1, 1] - navi[-1, 1])**2 + (g_truth[-1, 2] - navi[-1, 2])**2
            fde[j, 0] = math.sqrt(fd_error)
        ade[-1, 0] = np.sum(ade[0:len(agent), 0])/len(agent)
        ade_collect.append(ade)
        fde[-1, 0] = np.sum(fde[0:len(agent), 0]) / len(agent)
        fde_collect.append(fde)

    count = 0
    for f in fde_collect:
        np.savetxt(str(count)+'.txt', f)
        count+=1

    count = 0
    for a in ade_collect:
        np.savetxt(str(count)+'a.txt', a)
        count+=1


    # mod_x = mods_collect[3][5][0]
    # mod_y = mods_collect[3][5][1]
    # fig1 = mod_x.plot()
    # plt.show(block=True)
    # fig2 = mod_y.plot()
    # plt.show()


    # agents_present = agents_present[np.logical_and(agents_present[:, 5] < 497, agents_present[:, 5] > 299)]
    # agents_number = np.unique(agents_present[:, 0])
    #
    # agent = []
    # for i in agents_number:
    #     temp = agents_present[agents_present[:, 0] == i]
    #     temp = temp[:, (5, 1, 2)]
    #     temp = temp[temp[:, 0] % 7 == 6]
    #     temp[:, 0] = temp[:, 0] - 300
    #     agent.append(temp)
    #
    # observation = []
    # for i in agent:
    #     if len(i) < 8:
    #         observation.append(i)
    #         continue
    #     observation.append(i[0:8, :])
    #
    # time_span = agent[0][8:28, 0]


    # Even split of frame
    # test = navigation(observation, time_span)
    #
    # img = mpimg.imread('data/reference.jpg')
    # trej_ori = agent
    # trej_result = test[0]
    # for i in range(0, len(agent)):
    #     plt.imshow(img)
    #     plt.plot(trej_ori[i][:, 1], trej_ori[i][:, 2], "green")
    #     plt.plot(trej_result[i][0:len(trej_ori[i]), 1], trej_result[i][0:len(trej_ori[i]), 2], "red")
    #     plt.plot(trej_ori[i][0:8, 1], trej_ori[i][0:8, 2], "blue")
    #     plt.axis([0, 1409, 1916, 0])
    #     plt.show()
    #     for j in (0,1):
    #         temp_mod = test[1][i][j]
    #         fig2 = temp_mod.plot()
    #         matplotlib.pylab.show(block=True)
    #
    # result_collect = np.zeros((1, 4))
    # for i in range(0, len(agent)):
    #     agent_id = agents_number[i]
    #     id_col = np.full((len(trej_result[i]), 1), agent_id)
    #     temp_collect = np.column_stack((id_col, trej_result[i]))
    #     result_collect = np.row_stack((result_collect,temp_collect))
    # np.savetxt('result_56.txt', result_collect)

    # agent_sap = test[0][0]
    # plt.plot(agent_sap[:, 1], agent_sap[:, 2])
    # plt.show()
    #
    # k_mat = GPy.kern.RBF(input_dim=1, lengthscale=1)
    # k_lin = GPy.kern.Linear(1)
    # k = k_mat
    # k.plot()
    # plt.show()