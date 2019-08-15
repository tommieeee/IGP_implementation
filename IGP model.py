import numpy as np
import GPy
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.pylab

def path_generate(z, t):

    #generate trajactory for each agent
    #param x, y number of coordinates
    #param z agent index
    #param t number of time points
    #output trajactory coordinate (x,y) for each agent in a 3-dim numpy array
    mu, sigma = 0, 0.06
    x_coord = np.random.normal(mu, sigma, t)
    x_coord = x_coord.reshape(t, 1)
    y_coord = np.random.normal(0.2, sigma, 1)
    for j in range(2, t+1):
      y_coord = np.append(y_coord, np.random.normal(2*j/10, sigma, 1))
    y_coord = y_coord.reshape(t, 1)

    coord = np.column_stack((x_coord, y_coord))
    traj = coord

    for i in range(1,z):
        mu, sigma = i/10, 0.06
        x_coord = np.random.normal(mu, sigma, t)
        x_coord = x_coord.reshape(t, 1)
        y_coord = np.random.normal(0.2, sigma, 1)
        for j in range(2, t+1):
            y_coord = np.append(y_coord, np.random.normal(2*j/10, sigma, 1))
        y_coord = y_coord.reshape(t, 1)
        coord = np.column_stack((x_coord, y_coord))
        traj = np.dstack((traj, coord))
    return traj

def path_regress(traj):
    #input: trajectory of the agents
    #output: fitted gaussian regression for the trajactory of each agent
    models = []
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    time = np.array([range(traj.shape[0])])
    time = time.reshape(traj.shape[0], 1)
    for i in range(traj.shape[2]):
        coord = traj[:, :, i]
        x = coord[:, 0]
        x = x.reshape(len(x),1)
        y = coord[:, 1]
        y = y.reshape(len(y),1)
        m_x = GPy.models.GPRegression(time, x)
        m_x.optimize(messages=False)
        m_x.optimize_restarts(num_restarts = 50)
        m_y = GPy.models.GPRegression(time, y)
        m_y.optimize(messages=False)
        m_y.optimize_restarts(num_restarts = 50)
        m_xy = [m_x, m_y]
        models.append(m_xy)
    return models

def path_sample(mods, t, k, z):
    #sample trajactory from GP model
    #param mods: collection of GP model for each agent's trajectory
    #param t: number of time point need to be sampled
    #param k: number of samples
    #param z: number of agents
    testX = np.linspace(0, t, t+1).reshape(-1, 1)
    #posteriorTestY = model.posterior_samples_f(testX, full_cov=True, size=3)

    samples = np.empty((k,z,2,t+1))
    for count in range(k):
        coord = np.empty((z,2,t+1))
        for i in range(0,len(mods)):
            x_mod = mods[i][0]
            y_mod = mods[i][1]
            samp_x = x_mod.posterior_samples_f(testX, full_cov=True, size=1)
            samp_y = y_mod.posterior_samples_f(testX, full_cov=True, size=1)
            samp_coord = np.column_stack((samp_x[:,:,0], samp_y[:,:,0]))
            coord[i,:,:] = samp_coord.T
        samples[count, :, :, :] = coord


    return samples

def IA_potential(coord, a, h):
    result = 1
    for i in range(0,len(coord)-1):
        traj_i = coord[i,:,:]
        for j in range(i+1, len(coord)):
            traj_j = coord[j, :, :]
            for k in range(len(traj_j[0])):
                temp = 1-a*np.exp((-1/h**2)*np.linalg.norm(traj_i[:,k]-traj_j[:,k]))
                result = result*temp
    return result

def path_prediction(traj, z, t, n, a, h):
    #traj: original data
    #z: number of agents
    #t: time
    #n: number of samples
    mods = path_regress(traj)
    samples = path_sample(mods, t, n, z)
    max_index = 0
    max_potential = -1
    for i in range(0, n):
        curr = IA_potential(samples[i, :, :, :], a, h)
        if curr > max_potential:
            max_index = i
            max_potential = curr
    max_coord = samples[max_index, :, :, :]
    #for i in range(0, z):
     #  agent_i_ori = traj[:,:,i]
      #  new_agent_i = np.empty((2,t+1))
       ##new_agent_i[:, t] = agent_i[:,t+1]
    return  max_coord


if __name__ == '__main__':
    traj = path_generate(10, 20)
    plt.plot(traj[:, 0, :],traj[:, 1, :])
    plt.show()

    #coooo = path_sample(mods, 20, 1, 10)
    #pppp = coooo[0,:,:,:,]
    #pppp = pppp.reshape((21,2,10))
    #plt.plot(pppp[:, 0, :],pppp[:, 1, :])
    #plt.show()
    #potential = IA_potential(coooo[0, :, :, :],0.99, 0.05)


    # path_P = path_prediction(traj, 10, 20, 100, 0.99, 1)
    # for i in range(0,10):
    #     plt.plot(path_P[i, 0, :],path_P[i, 1, :])
    # plt.show()
    #
    # path_P = path_prediction(traj, 10, 20, 100, 0.99, 0.5)
    # for i in range(0,10):
    #     plt.plot(path_P[i, 0, :],path_P[i, 1, :])
    # plt.show()
    #
    path_P = path_prediction(traj, 10, 20, 100, 0.99, 0.1)
    for i in range(0,10):
        plt.plot(path_P[i, 0, :],path_P[i, 1, :])
    plt.show()
    #
    # path_P = path_prediction(traj, 10, 20, 100, 0.99, 0.01)
    # for i in range(0,10):
    #     plt.plot(path_P[i, 0, :],path_P[i, 1, :])
    # plt.show()
    #
    # path_P = path_prediction(traj, 10, 20, 100, 0.99, 0.0001)
    # for i in range(0,10):
    #     plt.plot(path_P[i, 0, :],path_P[i, 1, :])
    # plt.show()

    #plt.plot(path_P[1, 0, :], path_P[1, 1, :])
    #plt.show()

    #samp = path_sample(mods, 20, 1, 10)
    #samp = samp[0, :, :, :]
    #for i in range(0,10):
    #    plt.plot(samp[i, 0, :],samp[i, 1, :])
    #plt.show()

    mods = path_regress(traj)
    m = mods[0][0]
    display(m)
    fig = m.plot()
    matplotlib.pylab.show(block=True)
    display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_optimized'))
    plt.show()

    m.plot()



