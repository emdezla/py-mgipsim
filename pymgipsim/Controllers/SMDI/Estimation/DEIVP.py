import time
import importlib
from numba import njit, prange
import numpy as np
from typing import List
from scipy.stats import qmc
from matplotlib import pyplot as plt
import math
import matplotlib

class DifferentialEvolution:
    """ Observer class.

    """
    def __set_patient_params(self, parameter_array, no_meals, with_meal_estimation,
                           number_of_meal_param_coeff, meal_as_timestamped_array, patient_params, is_multiple_taud):
        """ Sets the patient parameters based on the parameter_array, which stores the estimated parameters
            (population of the differential evolution) in an array.

        Args:
            parameter_array:
            prev_tauds:
            no_meals:
            with_meal_estimation:
            prev_meals:
            number_of_meal_param_coeff:
            meal_as_timestamped_array:
            BW:
            CI:

        Returns:

        """
        self.Vg_array = parameter_array[:,3]
        p2_array = 1.0 / parameter_array[:,2]
        tau1_array = parameter_array[:,0]
        tau2_array = parameter_array[:,1]
        CI_array = patient_params.CI*np.ones((parameter_array.shape[0],))
        BW_array = patient_params.BW*np.ones((parameter_array.shape[0],))
        self.scalar_params_array = np.transpose(
            np.vstack((BW_array, CI_array, tau1_array, tau2_array, p2_array, self.Vg_array)))
        self.EGP_array = parameter_array[:, 4 + number_of_meal_param_coeff * no_meals::3]
        self.GEZI_array = parameter_array[:, 4 + number_of_meal_param_coeff * no_meals+1::3]
        self.SI_array = parameter_array[:, 4 + number_of_meal_param_coeff * no_meals+2::3]

        if with_meal_estimation:
            self.meal_array = np.array(
                parameter_array[:,
                4 + no_meals:4 + number_of_meal_param_coeff * no_meals])
        else:
            self.meal_array = np.multiply(np.ones((parameter_array.shape[0],meal_as_timestamped_array.shape[0])),meal_as_timestamped_array)
        if is_multiple_taud:
            self.taud_array = np.array(parameter_array[:,4:4 + no_meals])
        else:
            self.taud_array = np.multiply(np.ones((parameter_array.shape[0], meal_as_timestamped_array.shape[0])),
                        np.array(parameter_array[:,4:4 + no_meals]))


    def __init_device_arrays(self, parameter_array, fitnesses, observer=None):

        self.__set_patient_params(parameter_array,observer.no_meals,observer.with_meal_estimation,
                                  observer.number_of_meal_param_coeff,observer.meals,observer.patient_parameters,observer.is_multiple_taud)
        rate_of_appearance_time_array_full = np.linspace(
            observer.time[0] - observer.meal_times,
            observer.time[-1] - observer.meal_times,
            len(observer.time))
        rate_of_appearance_time_array_full[rate_of_appearance_time_array_full < 0.0] = 0.0

        self.rate_of_appearance_numba = np.zeros((observer.population_size, len(observer.interpolated_glucose_level)))
        self.mealImpulseResponseSum(rate_of_appearance_time_array_full,self.taud_array,self.Vg_array,self.meal_array,self.rate_of_appearance_numba)
        self.sections = np.asarray([np.inf])
        return fitnesses

    def run(self,observer):

        parameter_array = observer.parameter_array_init

        if observer.is_historical_init_state:
            self.x_init = np.asarray(observer.state_init)
        else:
            self.x_init = np.zeros((4,))

        fitnesses = np.zeros(observer.population_size, dtype=np.float64)
        # cohort = observer.generate_virtual_cohort(observer.population_size)

        glucose_array = observer.interpolated_glucose_level
        fitnesses = self.__init_device_arrays( parameter_array, fitnesses, observer)

        self.sim_glucose = np.zeros((observer.population_size,len(observer.insulins)))



        self.cost_function(self.scalar_params_array, self.rate_of_appearance_numba, np.zeros_like(observer.insulins,dtype=float),
                          observer.insulins, self.EGP_array, self.GEZI_array, self.SI_array,
                          self.x_init, fitnesses, glucose_array, self.sections, self.sim_glucose)
        # self.sim_glucose = self.rate_of_appearance_numba

        old_scalar_params = np.copy(self.scalar_params_array)
        for i in range(observer.max_iterations):

            new_pop = np.zeros_like(parameter_array)
            new_pop = self.__evolve_population(parameter_array, new_pop, 0.7,
                                               observer.lower_bounds,observer.upper_bounds,
                                               observer.basal_insulin,fitnesses)

            fitnesses = self.__init_device_arrays(new_pop, fitnesses, observer)
            fitness_eval = np.copy(fitnesses)

            start = time.time()
            new_glucose = np.copy(self.sim_glucose)
            self.cost_function(self.scalar_params_array, self.rate_of_appearance_numba,
                               np.zeros_like(observer.insulins, dtype=float),observer.insulins,
                              self.EGP_array, self.GEZI_array, self.SI_array,
                              self.x_init, fitness_eval, glucose_array, self.sections, new_glucose)
            # new_glucose = np.copy(self.rate_of_appearance_numba)
            new_fitness_local = np.copy(fitness_eval)
            constraint_ratio = self.selectPopulation(parameter_array,fitnesses,new_pop,new_fitness_local,observer.basal_insulin,
                                     observer.no_sections,observer.patient_parameters.CI[0],observer.number_of_meal_param_coeff,
                                     observer.no_meals, observer.steady_state_glucose_lower_bound, observer.steady_state_glucose_upper_bound,
                                  old_scalar_params, self.scalar_params_array, self.sim_glucose, new_glucose)
            # if observer.verbose:
                # print("Generation:",str(i)," Best:", str(np.min(fitnesses)), " Constaint ratio:",str(constraint_ratio))
        minidx = np.argmin(fitnesses)
        print("Cost achieved:", str(np.min(fitnesses)))
        #print(glucose_array)
        #print(self.sim_glucose[minidx])
        try:
            matplotlib.use("TkAgg")
        except:
            pass
        # plt.figure()
        # plt.plot(glucose_array)
        # plt.plot(self.sim_glucose[minidx])
        return parameter_array[minidx]

    @staticmethod
    @njit("float64[:,:](float64[:,:], float64[:,:], float64, float64[:],float64[:],float64,float64[:])")
    def __evolve_population(old_population, new_population, cr, lb, ub, basal_insulin, fitnesses):
        """ Updates the differential evolution population.

        Args:
            pop:
            pop2:
            cr:
            lb:
            ub:
            basal_insulin:
            fitnesses:

        Returns:

        """
        if not basal_insulin:
            basal_insulin = 16666.0
        npop, ndim = old_population.shape
        f = np.random.uniform(0.5, 3.0)
        v1 = np.argmin(fitnesses)
        for i in range(npop):

            # --- Vector selection ---
            v2, v3 = v1, v1
            while (v2 == i) or (v2 == v1):
                v2 = np.random.randint(npop)
            while (v3 == i) or (v3 == v2) or (v3 == v1):
                v3 = np.random.randint(npop)

            # --- Best 1 Mutation ---
            v = old_population[v1] + f * (old_population[v2] - old_population[v3])
            for param_i in range(ndim):
                if (v[param_i] < lb[param_i] or v[param_i] > ub[param_i]):
                    v[param_i] = np.random.rand()*(ub[param_i] - lb[param_i]) + lb[param_i]

            # --- Binomial Cross over ---
            co = np.random.rand(ndim)
            guaranteed_cr = np.random.randint(0, ndim)
            for j in range(ndim):
                if co[j] <= cr: #or j == guaranteed_cr
                    new_population[i, j] = v[j]
                else:
                    new_population[i, j] = old_population[i, j]

            # --- Forced crossing ---
            # j = np.random.randint(ndim)
            # pop2[i, j] = v[j]

        return new_population

    @staticmethod
    @njit(
        "(float64[:,:],float64[:,:],float64[:],float64[:],float64[:,:],float64[:,:],float64[:,:],float64[:],float64[:],float64[:],float64[:],float64[:,:])",
        parallel=True)
    def cost_function(scalar_params_cuda, rate_of_appearance_cuda, bolus_cuda, basal_cuda, EGP_cuda, GEZI_cuda,
                      SI_cuda, x_init, fitnesses, glucose_array, t_sections, sim_glucose):
        """ Calculates the fitness values of the individuals.

        Args:
            scalar_params_cuda:
            rate_of_appearance_cuda:
            bolus_cuda:
            basal_cuda:
            EGP_cuda:
            GEZI_cuda:
            SI_cuda:
            x_init:
            fitnesses:
            glucose_array:
            t_sections:

        Returns:

        """

        sim_length = glucose_array.shape[0] - 1
        for i in prange(0, scalar_params_cuda.shape[0]):
            patient = scalar_params_cuda[i]
            rate_of_appearance = rate_of_appearance_cuda[i]
            EGP = EGP_cuda[i]
            GEZI = GEZI_cuda[i]
            SI = SI_cuda[i]
            patient_glucose = sim_glucose[i]
            if x_init[0] < 10.0:
                Isc = basal_cuda[0] / patient[1]
                Ip = Isc
                Ieff = SI[0] * Ip
                G = EGP[0] / (GEZI[0] + Ieff)
            else:
                Isc = x_init[3]
                Ip = x_init[2]
                Ieff = x_init[1]
                G = x_init[0]
            patient_glucose[0] = G
            rmse = 0
            section = t_sections[0]
            EGPk = EGP[0]
            SIk = SI[0]
            GEZIk = GEZI[0]
            s = 0
            for k in range(sim_length):
                if k >= section:
                    EGPk = EGP[s + 1]
                    GEZIk = GEZI[s + 1]
                    SIk = SI[s + 1]
                    s = s + 1
                    section = t_sections[s]
                G = (- (GEZIk + Ieff) * G + EGPk + rate_of_appearance[k]) * 5 + G
                Ieff = patient[4] * (-Ieff + SIk * Ip) * 5 + Ieff
                Ip = (-Ip + Isc) / patient[3] * 5 + Ip
                Isc = (-Isc + (basal_cuda[k] + bolus_cuda[k]) / patient[1]) / patient[2] * 5 + Isc
                rmse = rmse + (G - glucose_array[k + 1]) ** 2
                patient_glucose[k+1] = G
            fitnesses[i] = math.sqrt(rmse / sim_length)

    @staticmethod
    @njit("(float64[:,:],float64[:,:],float64[:],float64[:,:],float64[:,:])", parallel=False)
    def mealImpulseResponseSum(rate_of_appearance_times, taud_array, Vg_array, meal_array,
                               rate_of_appearance_results):
        # print(rate_of_appearance_times)
        # print(rate_of_appearance_times.shape)
        # print(rate_of_appearance_results.shape)
        for y in range(rate_of_appearance_results.shape[0]):
            for x in range(rate_of_appearance_results.shape[1]):
                ra_current = 0.0
                no_meals = meal_array.shape[1]
                meals = meal_array[y]
                tauds = taud_array[y]
                time_array = rate_of_appearance_times[x]
                Vg = Vg_array[y]
                for m in range(no_meals):
                    d_temp = -time_array[m] / tauds[m]
                    ds_temp = (800.0 * meals[m] / Vg) / (tauds[m] ** 2)
                    e_temp = math.exp(d_temp)
                    m1_temp = ds_temp * time_array[m] * e_temp
                    ra_current = ra_current + m1_temp
                rate_of_appearance_results[y, x] = ra_current

    @staticmethod
    @njit(
        "float64(float64[:,:], float64[:], float64[:,:], float64[:],float64,int32,float64,int32,int32,float64,float64,float64[:,:],float64[:,:],float64[:,:],float64[:,:])")
    def selectPopulation(old_pop, old_fitness, new_pop, new_fitness, basal_insulin, no_sections, CI,
                         number_of_meal_param_coeff, no_meals, basal_lower, basal_upper, old_scalar_params,
                         new_scalar_params, old_glucose, new_glucose):
        """ Implements nonlinear constraint on the basal glucose level based on:
            J. Lampinen, "A constraint handling approach for the differential evolution algorithm,
            " Proceedings of the 2002 Congress on Evolutionary Computation

        Args:
            old_pop:
            old_fitness:
            new_pop:
            new_fitness:
            basal_insulin:
            no_sections:
            CI:
            number_of_meal_param_coeff:
            no_meals:

        Returns:

        """

        if not basal_insulin:
            basal_insulin = 16666.0
        npop, ndim = old_pop.shape
        constraint_ratio = 0
        for i in range(npop):
            old_c = True
            old_g = 0.0
            new_c = True
            new_g = 0.0
            for section in range(no_sections):
                EGP = old_pop[i, 4 + number_of_meal_param_coeff * no_meals + section * 3]
                GEZI = old_pop[i, 4 + number_of_meal_param_coeff * no_meals + section * 3 + 1]
                SI = old_pop[i, 4 + number_of_meal_param_coeff * no_meals + section * 3 + 2]
                steady_state_glucose = EGP / (GEZI + SI / CI * basal_insulin)
                if (steady_state_glucose < basal_lower or steady_state_glucose > basal_upper):
                    old_c = False
                    old_g = old_g + np.abs(steady_state_glucose - (basal_lower + basal_upper) / 2) - (
                                basal_upper - basal_lower) / 2
                EGP = new_pop[i, 4 + number_of_meal_param_coeff * no_meals + section * 3]
                GEZI = new_pop[i, 4 + number_of_meal_param_coeff * no_meals + section * 3 + 1]
                SI = new_pop[i, 4 + number_of_meal_param_coeff * no_meals + section * 3 + 2]
                steady_state_glucose = EGP / (GEZI + SI / CI * basal_insulin)
                if (steady_state_glucose < basal_lower or steady_state_glucose > basal_upper):
                    new_c = False
                    new_g = new_g + np.abs(steady_state_glucose - (basal_lower + basal_upper) / 2) - (
                                basal_upper - basal_lower) / 2
            if old_c == True:
                constraint_ratio = constraint_ratio + 1
            if (old_c == False and new_c == True):
                old_pop[i, :] = new_pop[i, :]
                old_fitness[i] = new_fitness[i]
                old_glucose[i] = new_glucose[i]
            if (old_c == True and new_c == True and new_fitness[i] < old_fitness[i]):
                old_pop[i, :] = new_pop[i, :]
                old_fitness[i] = new_fitness[i]
                old_glucose[i] = new_glucose[i]
            if (old_c == False and new_c == False and new_g < old_g):
                old_pop[i, :] = new_pop[i, :]
                old_fitness[i] = new_fitness[i]
                old_glucose[i] = new_glucose[i]
        return constraint_ratio / npop

