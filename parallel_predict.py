from IGP_scene_prediction import navigation, data_clean, overall_plot
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

if __name__ == '__main__':
    agent, observation, time_span = data_clean(300, 540, 12)
    # result, mod = navigation(observation, time_span, 96, 0.01, 100)
    # overall_plot(agent, result)

    p = Pool(4)
    arguments = [(observation, time_span, 12, 0.01, 100),
                 (observation, time_span, 24, 0.01, 100),
                 (observation, time_span, 48, 0.01, 100),
                 (observation, time_span, 96, 0.01, 100)]
    result = p.starmap(navigation, arguments)
    p.close()
    p.join()



