import numpy as np
import math
from scipy.optimize import minimize


class ThirdProblem:

    def __init__(self, mars_heliocentric_longitude):
        self._mars_heliocentric_longitude = mars_heliocentric_longitude
        self._mars_heliocentric_latitude = None
        self._mars_geocentric_latitude = None

    def find_mars_heliocentric_latitude(self, mars_geocentric_latitude, mars_orbit_radius):
        self._mars_geocentric_latitude = mars_geocentric_latitude
        self._mars_heliocentric_latitude = np.arctan(np.tan(mars_geocentric_latitude)*(1-1/mars_orbit_radius))
        mars_heliocentric_latitude_in_degree = (180 / math.pi) * self._mars_heliocentric_latitude
        return mars_heliocentric_latitude_in_degree

    @staticmethod
    def func_to_minimize_euclidean_distance(params, args):
        euclidean_distance = []
        mars_orbit_inclination = params[0]
        mars_heliocentric_latitude = args[0]
        for i in range(len(mars_heliocentric_latitude)):
            distance = math.pow(math.sin(mars_orbit_inclination - mars_heliocentric_latitude[i]), 2)
            euclidean_distance.append(distance)
        return np.sum(euclidean_distance)

    def fit_circle_for_mars_orbit(self, initial_param):
        params = minimize(self.func_to_minimize_euclidean_distance, initial_param,
                          args=[self._mars_heliocentric_latitude], bounds=[(None, None)])
        return params['x']
