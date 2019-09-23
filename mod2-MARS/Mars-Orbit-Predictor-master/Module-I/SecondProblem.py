import numpy as np
import math
from scipy.optimize import minimize


class SecondProblem:

    def __init__(self, triangulation_index_pair,
                 triangulation_earth_heliocentric_longitude_in_radian,
                 triangulation_mars_geocentric_longitude_in_radian):

        self._triangulation_index_pair = triangulation_index_pair
        self._triangulation_mars_geocentric_longitude_in_radian = triangulation_mars_geocentric_longitude_in_radian
        self._triangulation_earth_heliocentric_longitude_in_radian = triangulation_earth_heliocentric_longitude_in_radian
        self._triangulation_pair_count = self._triangulation_index_pair.shape[0]/2
        self._mars_radius_list = []
        self._mars_angular_position_list_in_degree = []
        self._mars_angular_position_list_in_radian = []

    def find_mars_projection_on_ecliptic_plane(self):
        for i in range(self._triangulation_pair_count):
            # Solve linear equation Ax = b
            A = np.ones((2, 2))
            b = np.zeros((2, 1))
            theta1 = self._triangulation_earth_heliocentric_longitude_in_radian[i * 2, 0]
            theta2 = self._triangulation_earth_heliocentric_longitude_in_radian[i * 2 + 1, 0]
            alpha1 = self._triangulation_mars_geocentric_longitude_in_radian[i * 2, 0]
            alpha2 = self._triangulation_mars_geocentric_longitude_in_radian[i * 2 + 1, 0]
            b[0, 0] = np.sin(theta1) - np.tan(alpha1) * np.cos(theta1)
            b[1, 0] = np.sin(theta2) - np.tan(alpha2) * np.cos(theta2)
            A[0, 1] = -1 * np.tan(alpha1)
            A[1, 1] = -1 * np.tan(alpha2)
            # print A
            # print b
            solution = np.matmul(np.linalg.inv(A), b)
            y = solution[0, 0]
            x = solution[1, 0]
            # print solution
            r = np.sqrt(x * x + y * y)
            if x > 0 and y > 0:
                phi = math.atan(y / x)
            elif x > 0 > y:
                phi = 2 * math.pi - math.atan(np.abs(y) / x)
            elif y > 0 > x:
                phi = math.pi - math.atan(y / np.abs(x))
            else:
                phi = math.pi - math.atan(np.abs(y) / np.abs(x))
            phi_in_degree = (180 / math.pi) * phi
            # print r
            # print phi
            # print phi_in_degree
            self._mars_radius_list.append(r)
            self._mars_angular_position_list_in_degree.append(phi_in_degree)
            self._mars_angular_position_list_in_radian.append(phi)

        return self._mars_radius_list, self._mars_angular_position_list_in_degree

    @staticmethod
    def func_to_minimize_euclidean_distance(params, args):
        euclidean_distance = []
        orbit_radius = params[0]
        triangulated_radial_distance_list = args[0]
        for i in range(len(triangulated_radial_distance_list)):
            distance = math.pow(orbit_radius - triangulated_radial_distance_list[i], 2)
            euclidean_distance.append(distance)
        return np.sum(euclidean_distance)

    def fit_circle_for_mars_orbit(self, initial_param):
        params = minimize(self.func_to_minimize_euclidean_distance, initial_param,
                          args=[self._mars_radius_list], bounds=[(1, None)])
        return params['x']
