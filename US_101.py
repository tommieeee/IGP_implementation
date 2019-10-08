from IGP_scene_prediction import navigation, overall_plot
from US_101_Utility import data_clean,  ade_and_fde
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

if __name__ == '__main__':
    agent, observation, time_span, a_num= data_clean(300, 372, 2)
    # result = navigation(observation, time_span, 12, 0.01, 100, 1, 0.99, 12)
    # test_save = ade_and_fde(agent, result[0])

    #  single time span producing
    ade_collect = np.zeros((10, 4))
    for j in range(10):
        p = Pool(4)
        arguments = [(observation, time_span, 2, 0.01, 100, 1, 0.99, 12),
                     (observation, time_span, 72, 0.01, 100, 1, 0.99, 12),
                     (observation, time_span, 144, 0.01, 100, 1, 0.99, 12),
                     (observation, time_span, 288, 0.01, 100, 1, 0.99, 12)]
        result = p.starmap(navigation, arguments)
        p.close()
        p.join()

        ade_agent = np.zeros((61, 4))
        fde_agent = np.zeros((61, 4))
        for i in range(4):
            a_and_f = ade_and_fde(agent, result[i][0])
            ade_agent[:, i] = a_and_f[0][:, 0]
            fde_agent[:, i] = a_and_f[1][:, 0]
            ade_collect[j, i] = a_and_f[0][0, -1]

