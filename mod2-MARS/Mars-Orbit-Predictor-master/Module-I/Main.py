import os
import pandas as pd
import numpy as np
import math
import sys
import FirstProblem as fp
import SecondProblem as sp
import ThirdProblem as tp
import FourthProblem as fp

if __name__ == '__main__':
    mars_opposition_data = pd.read_csv('../data/01_data_mars_opposition.csv')
    # print(mars_opposition_data.values.shape)
    mars_heliocentric_longitude = mars_opposition_data.values[:, 3:7]
    # print(mars_heliocentric_longitude.shape)
    mars_mean_longitude = mars_opposition_data.values[:, 9:13]
    # print(mars_mean_longitude.shape)
    mars_heliocentric_longitude_in_degree = mars_heliocentric_longitude[:, 0:1] * 30 + \
                                            mars_heliocentric_longitude[:, 1:2] + \
                                            mars_heliocentric_longitude[:, 2:3] / 60.0 + \
                                            mars_heliocentric_longitude[:, 3:4] / 3600.0
    mars_heliocentric_longitude_in_rad = mars_heliocentric_longitude_in_degree * math.pi / 180.0
    # print(mars_heliocentric_longitude_in_degree)
    # print(mars_heliocentric_longitude_in_rad)
    mars_mean_longitude_in_degree = mars_mean_longitude[:, 0:1] * 30 + \
                                    mars_mean_longitude[:, 1:2] + \
                                    mars_mean_longitude[:, 2:3] / 60.0 + \
                                    mars_mean_longitude[:, 3:4] / 3600.0
    mars_mean_longitude_in_rad = mars_mean_longitude_in_degree * math.pi / 180.0

    # Solution for first problem
    
    print 'Loss function :: log(arithmetic mean) - log(geometric mean)'
    print 'Initial guess of x :: 1.2'
    print 'Initial guess of y :: 0.2'
    first_problem = fp.FirstProblem([1.2, 0.2],
                                 mars_mean_longitude_in_rad,
                                 mars_heliocentric_longitude_in_rad)
    x, y = first_problem.minimize_loss_function(0)
    print "Optimized parameter values :: " + 'x  = ' + str(x) + ' and y = ' + str(y)
    print 'Radius and angle(in degree) pair w.r.t. reference line'
    for index in range(mars_heliocentric_longitude_in_rad.shape[0]):
        phi, radius = first_problem.get_angular_position_and_radius_of_mars(x, y,
                                                                          mars_mean_longitude_in_rad[index],
                                                                          mars_heliocentric_longitude_in_rad[index])
        print radius, phi*180/math.pi
    
    print 'Loss function :: sample variance'
    print 'Initial guess of x :: 1.2'
    print 'Initial guess of y :: 0.2'
    first_problem = fp.FirstProblem([1.2, 0.2],
                                 mars_mean_longitude_in_rad,
                                 mars_heliocentric_longitude_in_rad)
    x, y = first_problem.minimize_loss_function(1)
    print "Optimized parameter values :: " + 'x  = ' + str(x) + ' and y = ' + str(y)
    print 'Radius and angle(in degree) pair w.r.t. reference line'
    for index in range(mars_heliocentric_longitude_in_rad.shape[0]):
        phi, radius = first_problem.get_angular_position_and_radius_of_mars(x, y,
                                                                          mars_mean_longitude_in_rad[index],
                                                                          mars_heliocentric_longitude_in_rad[index])
        print radius, phi*180/math.pi

