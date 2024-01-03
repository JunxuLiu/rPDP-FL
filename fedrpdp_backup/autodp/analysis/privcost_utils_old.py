import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from pynverse import inversefunc


from autodp import rdp_acct, rdp_bank, dp_acct, privacy_calibrator, utils
from autodp.mechanism_zoo import SubsampleGaussianMechanism, GaussianMechanism

def generate_rdp_orders():
    dense = 1.07
    alpha_list = [int(dense ** i + 1) for i in range(int(math.floor(math.log(1000, dense))) + 1)]
    alpha_list = np.unique(alpha_list)
    return alpha_list

def generate_sampling_probs():
    """ generate a series of sampling probs for the following curve fitting.
    """
    q_list = []
    tmp_list = [1,2,3,4,5,6,7,8,9,10]
    for mul in [0.1,0.01,0.001]:
        q_list.extend(list(map(lambda x: x * mul, tmp_list)))
    q_list = np.unique(q_list)
    return q_list


def epsilon(rdp_func, T, p, delta, orders=None):
    if orders is not None:
        optimal_order, epsilon = _ternary_search(lambda order: _apply_pdp_sgd_analysis(rdp_func, order, T, p, delta), options=orders, iterations=72)
    else:
        ## TODO: have not been updated
        optimal_order, epsilon = _ternary_search(lambda order: _apply_pdp_sgd_analysis(rdp_func, order, T, p, delta), left=1, right=512, iterations=100)
    return  optimal_order, epsilon

def _apply_pdp_sgd_analysis(rdp_func, order, T, p, delta):
    rdp = p * T * rdp_func(order) + (1-p)
    eps = rdp - math.log(delta) / (order - 1)
    return eps

def _ternary_search(f, options=None, left=1, right=512, iterations=72): # 利用三叉搜索树寻找最优order
    """Performs a search over a closed domain [left, right] for the value which minimizes f."""
    if options is not None:
        left, right = 0, len(options)-1
        for i in range(iterations):
            left_third = max(0, int(left + (right - left) / 3))
            right_third = min(int(right - (right - left) / 3), len(options)-1)
            # print(left_third, f(options[left_third]), right_third, f(options[right_third]))
            if f(options[left_third]) <= f(options[right_third]):
                if right_third == right:
                    break
                right = right_third
            else:
                if left_third == left:
                    break
                left = left_third
            # print(left, right)
        
        if left == right:
            opt_order = options[right]
            return opt_order, f(opt_order)
        elif right-left == 1:
            eps1 = f(options[left])
            eps2 = f(options[right])
            if eps1 < eps2:
                # print('eps1: ', eps1, 'eps2: ', eps2)
                # print(options[left], eps1)
                return options[left], eps1
            else:
                return options[right], eps2
        else:
            opt_order = options[int((left + right) / 2)]
            return opt_order, f(opt_order)

    else:
        for i in range(iterations):
            left_third = left + (right - left) / 3
            right_third = right - (right - left) / 3
            if f(left_third) < f(right_third):
                right = right_third
            else:
                left = left_third
        # print('==> ternary_search: ', i, left, right)
        return (left + right) / 2
    

class PrivCostEstimstor:
    def __init__(self, sigma, T, delta=1e5):
        self.sigma = sigma
        self.T = T
        self.delta = delta

    def estimator(self, est_object='probs', alpha_list=None, q_list=None, plot=False):
        if alpha_list is None:
            alpha_list = generate_rdp_orders()

        if q_list is None:
            q_list = generate_sampling_probs()

        examples = self._generate_examples(q_list, alpha_list)
        fit_func, prop = self._curve_fitting(q_list, examples, func_type='quad')
        if plot:
            ## TODO plot the figure
            # print('the plot module has not been implemented yet.')
            self._plot(q_list, examples, fit_func, prop)

        if est_object == 'privcost':
            return fit_func

        elif est_object == 'probs':
            inv_fit_func = inversefunc(fit_func) # inv_fit_func(y)
            return inv_fit_func

    def _generate_examples(self, q_list, alpha_list, mech='gauss_poisson'):
        opt = []
        if mech == 'gauss_poisson':
            # print(self.sigma, self.T)
            mech = GaussianMechanism(sigma=self.sigma)
            for q in q_list:
                acct = rdp_acct.anaRDPacct()
                acct.compose_poisson_subsampled_mechanisms(mech.RenyiDP, q)
                acct.build_zeroth_oracle()
                opt_order, est_priv_cost = epsilon(acct.RDPs[0], T=self.T, p=1.0, delta=self.delta, orders=alpha_list)
                # print(q, opt_order, est_priv_cost)
                opt.append(est_priv_cost)
                del acct

        else:
            #TODO: more types of mechanisms
            pass
            
        return opt

    def _generate_examples2(self, q_list, alpha_list, mech='gauss_poisson'):
        opt = []
        if mech == 'gauss_poisson':
            # print(self.sigma, self.T)
            mech = GaussianMechanism(sigma=self.sigma)
            for q in q_list:
                acct = rdp_acct.anaRDPacct()
                acct.compose_poisson_subsampled_mechanisms(mech.RenyiDP, q)
                acct.build_zeroth_oracle()
                rdp_list = [acct.RDPs[0](alpha)*self.T for alpha in alpha_list]
                eps_list = [rdp - math.log(self.delta) / (alpha - 1) for (alpha, rdp) in zip(alpha_list, rdp_list)]
                idx_opt = np.nanargmin(eps_list)  # Ignore NaNs
                est_priv_cost = eps_list[idx_opt]
                # print(q, alpha_list[idx_opt], est_priv_cost)
                # est_priv_cost = epsilon(acct.RDPs[0], T=self.T, p=1.0, delta=self.delta, orders=alpha_list)
                opt.append(est_priv_cost)
                del acct

        else:
            #TODO: more types of mechanisms
            pass
            
        return opt

    def _curve_fitting(self, xdata, ydata, func_type = 'quad'):
        '''
        Input: 
        Output: None (fit_func) 
        '''
        if func_type == 'quad':
            func = lambda x, a, b, c: a + b*x + c*x**2
            popt, pcov = curve_fit(func, xdata, ydata)
            # print(popt) #output: array([ 2.55423706, 1.35190947, 0.47450618])
            r2 = r2_score(func(xdata, *popt), ydata)
            print('r2 score of the curve fitting.', r2)
            return lambda x: popt[0] + popt[1]*x + popt[2]*x**2, popt

        elif func_type == 'exp':
            func = lambda x, a, b, c: a * np.exp(b*x) + c
            popt, pcov = curve_fit(func, xdata, ydata)
            # print(popt) #output: array([ 2.55423706, 1.35190947, 0.47450618])
            r2 = r2_score(func(xdata, *popt), ydata)
            print('r2 score of the curve fitting.', r2)
            return lambda x: popt[0] * np.exp(-popt[1] * x) + popt[2], popt
    
    def _plot(self, x, y, fit_curve, popt):
        font = {'family': 'times',
                'weight': 'bold',
                'size': 13}

        matplotlib.rc('font', **font)
        my_colors = list(mcolors.TABLEAU_COLORS.keys())

        plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(x, y, color=my_colors[0], linewidth=1)
        plt.plot(x, fit_curve(x), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

        #Labels
        plt.title(r'Subsampled gaussian with $\sigma=10.0, \delta=1e-5$\n(Quadratic Function Fitting)')
        plt.xlabel(r'sampling ratio $\alpha$')
        plt.ylabel(rf'$\epsilon({self.delta})$')
        plt.legend()
        leg = plt.legend()  # remove the frame of Legend, personal choice
        leg.get_frame().set_linewidth(0.0) # remove the frame of Legend, personal choice
        #leg.get_frame().set_edgecolor('b') # change the color of Legend frame
        #plt.show()")
        plt.grid(True)
        plt.savefig("../results/q_opteps_scatter.pdf", bbox_inches='tight')
        plt.show()


class PrivCostAnalyser:
    def __init__(self, sigma, steps, num_users, budgets, p_max=1.0, cprob=1.0, delta=1e-5):
        self.sigma = sigma
        self.steps = steps
        # self.data_sampling_prob_list = dprobs
        self.num_users = num_users
        self.p_max = p_max # client.batch_size / len(client.train_data)
        self.client_sampling_prob = cprob
        self.delta = delta

        self.gauss_mech = GaussianMechanism(sigma=self.sigma)
        self.alpha_list = generate_rdp_orders()
        self.RDPs_filter = np.array(budgets)
        self.RDPs_odometer = np.zeros_like((num_users, self.alpha_list), float)
        self.round3 = lambda f: round(f, 3)

        print(self.gauss_mech.RenyiDP, p_max)
        acct = rdp_acct.anaRDPacct()
        acct.compose_poisson_subsampled_mechanisms(self.gauss_mech.RenyiDP, self.p_max)
        self._cache = {self.round3(self.p_max):np.array([acct.RDPs[0](alpha)*self.steps for alpha in self.alpha_list])}
        del acct
    
    def _check_leftover_bgts(self):
        accum_loss = []
        for user in range(self.num_users):
            opt_eps = min(self.RDPs_odometer[user] - math.log(self.delta) / (self.alpha_list - 1))
            accum_loss.append(opt_eps)
        return np.array(accum_loss)
        # count = np.sum(calc_min_eps() > self.RDPs_filter)
        # return count

    def update_priv_cost(self, dprobs, T=1, improved_bound_flag = True):
        if improved_bound_flag:
            for user, prob in enumerate(dprobs):
                assert prob >= 0.0 and prob <= 1.0, "the data sampling probability must be [0,1]."
                if self._cache.get(self.round3(prob)) is None:
                    acct = rdp_acct.anaRDPacct()
                    acct.compose_poisson_subsampled_mechanisms(self.gauss_mech.RenyiDP, prob)
                    self._cache[self.round3(prob)] = np.array([acct.RDPs[0](alpha)*self.steps for alpha in self.alpha_list])
                    del acct

                RDPs_int = self._cache[self.round3(prob)]*T
                self.RDPs_odometer[user] += RDPs_int
        else:
            ## TODO: to be updated
            acct.compose_poisson_subsampled_mechanisms1(self.gauss_mech.RenyiDP, prob)
        
        tot_active_points = np.sum(self._check_leftover_bgts() <= self.RDPs_filter)
        print("total activate points: ", tot_active_points)
    

    # def calc_priv_cost(self, dprobs, steps, improved_bound_flag = True):
    #     if improved_bound_flag:
    #         final_priv_cost = []
            
    #         for prob in dprobs:
    #             assert prob >= 0.0 and prob <= 1.0, "the data sampling probability must be [0,1]."
    #             acct = rdp_acct.anaRDPacct()
    #             acct.compose_poisson_subsampled_mechanisms(self.gauss_mech.RenyiDP, prob)
    #             opt_order, tot_priv_cost = epsilon(acct.RDPs[0], T=steps, p=self.client_sampling_prob, rdp_odometer=self.rdp_odometer, orders=self.alpha_list, tgt_delta=self.delta)
    #             # tot_priv_cost = self._epsilon(acct.RDPs[0])
    #             # print('data sampling prob: ', prob, ' tot_priv_cost(optimal):', tot_priv_cost)
    #             final_priv_cost.append(tot_priv_cost)
    #             del acct

    #         return final_priv_cost

    #     else:
    #         ## TODO: to be updated
    #         acct.compose_poisson_subsampled_mechanisms1(self.gauss_mech.RenyiDP, prob)