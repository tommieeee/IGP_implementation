import numpy as np
import GPy
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.pylab
import math


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
    k = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    mod_x = GPy.models.GPRegression(time, x_dir, k)
    mod_x.optimize(messages=False)
    mod_x.optimize_restarts(num_restarts = 30)

    time = traj[:, 0].reshape(len(traj[:, 0]), 1)
    k = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    y = traj[:, 2]
    y = y.reshape(len(y), 1)
    m_y = GPy.models.GPRegression(time, y, k)
    m_y.optimize(messages=False)
    m_y.optimize_restarts(num_restarts = 30)
    m_xy = [mod_x, m_y]

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
        traj_i = coord[i,:,:]
        for j in range(i+1, len(coord)):
            traj_j = coord[j, :, :]
            for k in range(len(traj_j[0])):
                temp = 1-alpha*np.exp((-1/h**2)*np.linalg.norm(traj_i[:,k]-traj_j[:,k]))
                result = result*temp
    return result



def path_sample(mods, time_axis):
    sample_path = np.empty((len(mods), len(time_axis), 2))
    for i in range(len(mods)):
        agt_mod = mods[i]
        x_mod = agt_mod[0]
        y_mod = agt_mod[1]
        samp_x = x_mod.posterior_samples_f(time_axis, full_cov=True, size=1).reshape((-1, 1))
        samp_y = y_mod.posterior_samples_f(time_axis, full_cov=True, size=1).reshape((-1, 1))
        coord = np.column_stack((samp_x, samp_y))
        sample_path[i, :, :] = coord
    return sample_path


def path_prediction(mods, t_points, samp_num):
    """
    find the predicted path of the overall crowds with the max interaction potential
    :param mods: GP model of each agent and provisional ROBOT
    :param t_points: time points desired to sample
    :param samp_num: number of samples want to look at
    :return: trajectory with the max interaction potential at the time span
    """
    h = 1
    alpha = 0.99
    time_points = t_points.reshape((-1,1))
    sample_collection = np.empty((samp_num, len(mods), len(time_points), 2))
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
    curr_traj = agt[agt[:, 0] <= t_point]
    return curr_traj

def navigation(obsv, t_span, destination):
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
    curr_span = np.array([])
    p_robot_traj = destination
    for t_point in t_span:
        mods = []
        p_robot_mod = agent_regress(p_robot_traj)
        mods.append(p_robot_mod)
        curr_span = np.append(curr_span, t_point)
        for agt in obsv:
            agt_traj = time_match(agt, t_point)
            if len(agt_traj) == 0:
                continue
            agt_mod = agent_regress(agt_traj)
            mods.append(agt_mod)
        optimized_path = path_prediction(mods, curr_span, 100)
        robot_path = np.column_stack((curr_span, optimized_path[0, :, :]))
        next_location = robot_path[-1, :]
        p_robot_traj = np.vstack((p_robot_traj, next_location))

    return p_robot_traj, mods



if __name__ == '__main__':
    mat = np.loadtxt("ewap_dataset\seq_eth\obsmat.txt", dtype="float")

    count = 0
    max_count = 0
    max_index = -1
    for i in range(1, len(mat[:, 0])):
        if mat[i, 0] == mat[i-1, 0]:
            count = count+1
            if count > max_count:
                max_index = i
                max_count = count
        else:
            count = 0
    index = []
    for i in range(0, len(mat[:, 1])):
        if mat[i, 0] >= 10251 and mat[i, 0] <= 10527:
            index.append(i)
    target_time = mat[index, :]


    final_result = []
    for ch_one in range(0,10):

        agents = []
        for i in range(260, 270):
            index = []
            for j in range(0, len(target_time[:, 1])):
                if target_time[j, 1] == i:
                    index.append(j)
            aj = target_time[index, :]
            agents.append(aj[:, (0, 2, 4)])

        robot = agents[ch_one]
        del(agents[ch_one])
        way_point = robot[(0, 1, 10, 20, -2, -1), :]
        way_point = np.reshape(way_point,(6, 3))
        time_span = np.delete(robot, (0, 1, 10, 20, len(robot)-2, len(robot)-1), axis = 0)
        test, mods = navigation(agents, time_span[:, 0], way_point)

        test = test[test[:,0].argsort()]


        time_points = time_span[:, 0]
        collect = []
        collect2 = []
        for i in time_points:
            a = test[test[:, 0] == i, :]
            ori = robot[robot[:, 0] == i, :]
            for j in range(0, len(agents)):
                d = agents[j]
                b = d[d[:, 0] == i, :]
                if b.size > 0 and a.size > 0:
                    c = ((a[0 ,1]-b[0 ,1])**2 + (a[0 ,2]-b[0 ,2])**2)
                    dist = math.sqrt(c)
                    collect.append(dist)
                    c = ((ori[0 ,1]-b[0 ,1])**2 + (ori[0 ,2]-b[0 ,2])**2)
                    dist = math.sqrt(c)
                    collect2.append(dist)
        collect = np.array(collect)
        robot_min = collect.min()
        collect2 = np.array(collect2)
        ori_min = collect2.min()
        record = [robot_min, ori_min]

        length = 0
        for i in range(1, len(test)):
            a = test[i, :]
            b = test[i-1, :]
            c = (a[1]-b[1])**2 + (a[2]-b[2])**2
            dist = math.sqrt(c)
            length += dist

        length_ori = 0
        for i in range(1, len(robot)):
            a = robot[i, :]
            b = robot[i-1, :]
            c = ((a[1]-b[1])**2 + (a[2]-b[2])**2)
            dist = math.sqrt(c)
            length_ori += dist

        record2 = [length, length_ori]

        result = [record, record2]
        final_result.append(result)



