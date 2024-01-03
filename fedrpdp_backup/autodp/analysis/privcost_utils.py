import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from pynverse import inversefunc, piecewise

from autodp import rdp_acct, utils
from autodp.mechanism_zoo import GaussianMechanism

import numpy as np

# provided PDP recipes 

def MultiLevels(n_levels, ratios, values, size):
    assert abs(sum(ratios) - 1.0) < 1e-6, "the sum of `ratios` must equal to one."
    assert len(ratios) == len(values) and len(ratios) == n_levels, "both the size of `ratios` and the size of `values` must equal to the value of `n_levels`."

    tgt_epsilons = [0]*size
    pre = 0
    for i in range(n_levels-1):
        l = int(size * ratios[i])
        tgt_epsilons[pre: pre+l] = [values[i]]*l
        pre = pre+l
    l = size-pre
    tgt_epsilons[pre:] = [values[-1]]*l
    return np.array(tgt_epsilons)

def MixGauss(ratios, means_and_stds, size):
    assert abs(sum(ratios) - 1.0) < 1e-6, "the sum of `ratios` must equal to one."
    assert len(ratios) == len(means_and_stds), "the size of `ratios` and `means_and_stds` must be equal."

    tgt_epsilons = []
    pre = 0
    for i in range(size):
        # random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]) # 掷骰子（多项式分布）
        dist_idx = np.argmax(np.random.multinomial(1, ratios))
        value = np.random.normal(loc=means_and_stds[dist_idx][0], scale=means_and_stds[dist_idx][1])
        tgt_epsilons.append(value)
    
    return np.array(tgt_epsilons)

def Gauss(mean_and_std, size):
    return np.random.normal(loc=mean_and_std[0], scale=mean_and_std[1], size=size)


def Pareto(shape, mean, scale, size):
    return (np.random.pareto(shape, size) + mean) * scale

BOUNDED_BUDGET_FUNC = lambda x, minimum, maximum: min(max(x, minimum), maximum)

def generate_rdp_orders():
    dense = 1.07
    alpha_list = [int(dense ** i + 1) for i in range(int(math.floor(math.log(1000, dense))) + 1)]
    alpha_list = np.unique(alpha_list)
    return alpha_list

# def generate_sampling_probs():
#     """ generate a series of sampling probs for the following curve fitting.
#     """
#     q_list = []
#     tmp_list = [1,2,3,4,5,6,7,8,9,10]
#     for mul in [0.1,0.01,0.001,0.0001]:
#         q_list.extend(list(map(lambda x: x * mul, tmp_list)))
#     q_list.extend(list(map(lambda x: 0.9 + x * 0.01, tmp_list))) 
#     q_list = sorted(np.unique(q_list))
#     return q_list

def epsilon(rdp_func, k, T, p, delta, orders=None):
    if orders is not None:
        optimal_order, epsilon = _ternary_search(lambda order: _apply_pdp_sgd_analysis(rdp_func, order, k, T, p, delta), options=orders, iterations=72)
    else:
        ## TODO: have not been updated
        optimal_order, epsilon = _ternary_search(lambda order: _apply_pdp_sgd_analysis(rdp_func, order, k, T, p, delta), left=1, right=512, iterations=100)
    return  optimal_order, epsilon

def _apply_pdp_sgd_analysis(rdp_func, order, k, T, p, delta):
    '''
    params::
        rdp_func: rdp budget curve, usage: rdp_func(order)
        order: rdp order
        k: inner iterations (data-level subsampling and DPSGD)
        T: outer communication rounds (client-level subsampling)
        p: (uniform) client sampling ratio
        delta: target dp parameter `delta`
    '''
    if T == 0:
        rdp = rdp_func(order)*k # for the case of DPSGD
    elif T > 0 and p == 1.: # p=1. : client sample ratio == 1.0
        rdp = rdp_func(order)*k*T # full clients participation
    else:
        rdp = _two_stage_subsampled_gauss_rdp_analysis(rdp_func, order, k, T, p)
        
    eps = rdp + np.log(1/delta) / (order-1)
    return eps

def _two_stage_subsampled_gauss_rdp_analysis(rdp_func, order, k, T, p):

    log_term_1 = np.log(1-p)
    log_term_2 = np.log(p) + (order-1) * k * rdp_func(order)
    rdp_stage1 = 1.0*utils.stable_logsumexp([log_term_1, log_term_2])/(order-1)
    rdp_stage2 = rdp_stage1 * T
    
    return rdp_stage2

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
    
class PrivCostEstimator:
    def __init__(self, sigma, inner_iters, outer_iters=0, outer_ratio=1.0, delta=1e-5):
        self.sigma = sigma # the ratio of std / l2_norm_clip
        self.k = inner_iters
        self.T = outer_iters
        self.p = outer_ratio
        self.delta = delta

    def estimator(self, est_object='probs', alpha_list=None, q_list=None, plot=False):
        if alpha_list is None:
            alpha_list = generate_rdp_orders()

        if q_list is None:
            # q_list = generate_sampling_probs()
            q_list = np.concatenate([np.arange(0, 1.0, 0.01), np.arange(0, 0.1, 0.001), [1.0]])
            q_list = sorted(np.unique(q_list))[1:] # delete q = 0.0
            
        examples, upperbound, lowerbound = self._generate_examples(q_list, alpha_list)
        fit_func, prop = self._curve_fitting(q_list, examples, func_type='exp')
        
        print('check the valid scope of privacy budget: ', upperbound, lowerbound) 
        if plot:
            ## TODO plot the figure
            # print('the plot module has not been implemented yet.')
            self._plot(q_list, examples, fit_func, prop)

        if est_object == 'privcost':
            return fit_func

        elif est_object == 'probs':
            ## TODO conditioned fit func, i.e., if the input is larger than the maximum, always output 1.0
            # inv_fit_func = lambda x: float(inversefunc(fit_func, accuracy=6)(x)) if x < maximum else 1.0 # inv_fit_func(y)
            # pw= lambda x: piecewise(x,[x<maximum, x>=maximum],[lambda x: fit_func(x), lambda x: x**2, lambda x: x+6])
            # inv_fit_func = inversefunc(fit_func, accuracy=6, domain=[0.0,1.0])
            def inv_fit_func(x):
                
                if x >= upperbound:
                    return 1.0
                elif x <= lowerbound:
                    return 0.0
                else:
                    return inversefunc(fit_func)(x)
            return inv_fit_func

    def _generate_examples(self, q_list, alpha_list, mech='gauss_poisson'):
        """
        return: opt_budget, upperbound, lowerbound
        """
        opt = []
        maximum_budget, minimum_budget = None, None # when the sampling ratio = 1.0, min(q_list)
        if mech == 'gauss_poisson':
            # print(self.sigma, self.T)
            mech = GaussianMechanism(sigma=self.sigma)
            for i,q in enumerate(q_list):
                if np.abs(q - 0.0) < 1e-5:
                    opt.append(0)
                    continue

                acct = rdp_acct.anaRDPacct()
                acct.compose_poisson_subsampled_mechanisms(mech.RenyiDP, q)
                acct.build_zeroth_oracle()
                opt_order, est_priv_cost = epsilon(acct.RDPs[0], k=self.k, T=self.T, p=self.p, delta=self.delta, orders=alpha_list)
                opt.append(est_priv_cost)

                if np.abs(q - 1.0) < 1e-5:
                    # print(r'the largest value of epsilon: ', q, opt_order, est_priv_cost)
                    maximum_budget = est_priv_cost
                
                if np.abs(q - min(q_list)) < 1e-5:
                    # print(r'the smallest value of epsilon: ', q, opt_order, est_priv_cost)
                    minimum_budget = est_priv_cost

                del acct

            # if 1.0 is not exist in q_list, don't forget it !
            if maximum_budget is None:
                opt_order, maximum_budget = epsilon(mech.RenyiDP, k=self.k, T=self.T, p=self.p, delta=self.delta, orders=alpha_list)

        else:
            #TODO: more types of mechanisms
            pass
        print('maximum budget:', maximum_budget, 'minimum budget:', minimum_budget)
        return opt, maximum_budget, minimum_budget

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

    def _curve_fitting(self, xdata, ydata, func_type = 'exp'):
        '''
        Input: 
        Output: None (fit_func) 
        '''
        # if func_type == 'quad':
        #     xdata = np.array(xdata, dtype=np.float32)
        #     ydata = np.array(ydata, dtype=np.float32)
        #     func = lambda x, a, b, c: a*x**2 + b*x + c
        #     popt, pcov = curve_fit(func, xdata, ydata)
        #     # print(popt) #output: array([ 2.55423706, 1.35190947, 0.47450618])
        #     r2 = r2_score(func(xdata, *popt), ydata)
        #     print('r2 score of the curve fitting.', r2)
        #     # return lambda x: popt[0]*x**2 + popt[1]*x + popt[2], popt
        #     return lambda x: func(x, *popt), popt
        # 
        # elif func_type == 'exp':
        # 
        xdata = np.array(xdata, dtype=np.float32)
        ydata = np.array(ydata, dtype=np.float32)
        print(xdata)
        print(ydata)
        func = lambda x, a, b, c: a * np.exp(b*x) + c
        popt, pcov = curve_fit(func, xdata, ydata)
        r2 = r2_score(func(xdata, *popt), ydata)
        print('r2 score of the curve fitting.', r2)
        return lambda x: func(x, *popt), popt
    
    def _plot(self, x, y, fit_curve, popt):
        font = {'weight': 'bold', 'size': 13}

        # matplotlib.rc('font', **font)
        # my_colors = list(mcolors.TABLEAU_COLORS.keys())

        plt.figure(num=1, figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(x, y, linewidth=1)
        plt.plot(x, [fit_curve(elem) for elem in x], 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

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
        plt.savefig("q_opteps_scatter.pdf", bbox_inches='tight')
        plt.show()

    def _plot_inverse(self, x, y, inv_fit_curve, popt):
        font = {'weight': 'bold', 'size': 13}

        matplotlib.rc('font', **font)
        my_colors = list(mcolors.TABLEAU_COLORS.keys())

        plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(x, y, color=my_colors[0], linewidth=1)
        plt.plot(x, inv_fit_curve(x), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

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
        plt.savefig("q_opteps_scatter.pdf", bbox_inches='tight')
        plt.show()
        plt.close('all')


class PrivCostAnalyser:
    def __init__(self, sigma, inner_iters, outer_iters, inner_ratios, outer_ratio=1.0, delta=1e-5):
        self.sigma = sigma
        self.inner_iters = inner_iters
        self.outer_iters = outer_iters
        # self.data_sampling_prob_list = dprobs
        self.inner_ratios = inner_ratios
        self.outer_ratio = outer_ratio
        self.delta = delta

        # self.gauss_mech = GaussianMechanism(sigma=self.sigma)
        self.alpha_list = generate_rdp_orders()

        ## TODO: rdp_filters and rdp_odometers
        
        self.round3 = lambda f: round(f, 3)

        # acct = rdp_acct.anaRDPacct()
        # acct.compose_poisson_subsampled_mechanisms(self.gauss_mech.RenyiDP, self.p_max)
        # self._cache = {self.round3(self.p_max):np.array([acct.RDPs[0](alpha)*self.steps for alpha in self.alpha_list])}
        # del acct
    
    def compute_priv_cost(self, range="minmax"):
        gauss_mech = GaussianMechanism(sigma=self.sigma)
        priv_cost_list = []
        if range == "all":
            for q in np.unique(self.inner_ratios):
                acct = rdp_acct.anaRDPacct()
                acct.compose_poisson_subsampled_mechanisms(gauss_mech.RenyiDP, q)
                acct.build_zeroth_oracle()
                opt_order, priv_cost = epsilon(acct.RDPs[0], k=self.inner_iters, T=self.outer_iters, p=self.outer_ratio, delta=self.delta, orders=self.alpha_list)
                priv_cost_list.append(priv_cost)
            
        elif range == "minmax":
            maxq = max(self.inner_ratios)
            acct = rdp_acct.anaRDPacct()
            acct.compose_poisson_subsampled_mechanisms(gauss_mech.RenyiDP, maxq)
            acct.build_zeroth_oracle()
            opt_order, min_priv_cost = epsilon(acct.RDPs[0], k=self.inner_iters, T=self.outer_iters, p=self.outer_ratio, delta=self.delta, orders=self.alpha_list)
            priv_cost_list.append(min_priv_cost)
            # print(q, eps_list[idx_opt], self.alpha_list[idx_opt])
            del acct

            minq = min(self.inner_ratios)
            acct = rdp_acct.anaRDPacct()
            acct.compose_poisson_subsampled_mechanisms(gauss_mech.RenyiDP, minq)
            acct.build_zeroth_oracle()
            opt_order, max_priv_cost = epsilon(acct.RDPs[0], k=self.inner_iters, T=self.outer_iters, p=self.outer_ratio, delta=self.delta, orders=self.alpha_list)
            priv_cost_list.append(max_priv_cost)
            # print(q, eps_list[idx_opt], self.alpha_list[idx_opt])
            del acct

        return priv_cost_list
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