import math
import numpy as np
from scipy.optimize import minimize


class FourthProblem:

    def __init__(self, mars_triangulation_radius_list, mars_orbital_inclination_in_radian):
        self._mars_triangulation_radius_list = mars_triangulation_radius_list
        self._mars_orbital_inclination_in_radian = mars_orbital_inclination_in_radian
        self._mars_triangulated_radius_orbital_plane = None

    def find_mars_3d_location_in_orbital_plane(self):
        self._mars_triangulated_radius_orbital_plane = [radius / math.cos(self._mars_orbital_inclination_in_radian)
                                                        for radius in self._mars_triangulation_radius_list]
        return self._mars_triangulated_radius_orbital_plane

    @staticmethod
    def func_to_minimize_euclidean_distance(params, args):
        euclidean_distance = []
        orbit_radius = params[0]
        triangulated_radial_distance_list = args[0]
        for i in range(len(triangulated_radial_distance_list)):
            distance = math.pow(orbit_radius - triangulated_radial_distance_list[i], 2)
            euclidean_distance.append(distance)
        return np.sum(euclidean_distance)

    @staticmethod
    def _find_loss(x1, x2):
        loss_list = []
        for i in range(len(x1)):
            loss = math.pow(x1[i] - x2, 2)
            loss_list.append(loss)
        return np.sum(loss_list)

    def fit_circle_to_mars_orbital_plane(self, initial_param):
        params = minimize(self.func_to_minimize_euclidean_distance, initial_param,
                          args=[self._mars_triangulated_radius_orbital_plane], bounds=[(1, None)])
        best_fit_radius = params['x']
        total_loss = self._find_loss(self._mars_triangulated_radius_orbital_plane, best_fit_radius)
        return best_fit_radius, total_loss

    # def fit_ellipse_to_mars_orbital_plane(self):
