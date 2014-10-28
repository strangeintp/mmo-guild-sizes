'''
Created on May 8, 2014

@author: Vince Kane

'''

from random import random
from random import randint
from scipy.optimize import minimize
from math import log
from math import exp
import numpy as np
from utility import cdf
from utility import fromCDF
import collections
import traceback

VERBOSE = True
def verbose(stuff):
    if VERBOSE:
        print(stuff)

def loglikelihood_binned(B, H, b_min, alpha):
    '''
    B = list of bin boundaries, b0 > 1
    H = list of counts h_i in each bin, b_i <= x_i < b_i+1
    b_min = bin containing the presumed lower bound x_min
    alpha = power law exponent
    ********THIS FUNCTION ONLY WORKS WITH ALPHA > 1 !!!************
    '''
    #find index of b_min in B
    idx_min = B.index(b_min)
    sum_terms = 0
    for i in range(idx_min, len(B)-1):
        sum_terms += H[i] * (log(B[i]**(1-alpha) - B[i+1]**(1-alpha)) )
    n = sum(H[idx_min:])
    sum_terms += n*-(alpha-1)*log(b_min)
    return sum_terms

def obj_loglikelihood_binned(alpha, bins, bmin, h):
    return -loglikelihood_binned(bins, h, bmin, alpha)

def power(params, x):
    return params[1]*x**-params[0]

def linear(params, x):
    return x*params[0] + params[1]

def exponential(params, x):
    return params[1]*exp(-params[0]*x)

def lognormal(params, x):
    return (params[2]/x)*exp(-(log(x)-params[0])**2/(2*params[1]**2))

def error_sumsq(params, eval_func, bins, bmin, h):
    idx = bins.index(bmin)
    sum_terms = 0
    for i in range(idx, len(h)):
        sum_terms += (h[i] - eval_func(params, bins[i]))**2
    return sum_terms

def KS(params, eval_func, bins, bmin, h):
    idx = bins.index(bmin)
    h_hat = [eval_func(params, b) for b in bins[idx:len(h)]]
    try:    
        cdf_h_hat = cdf(h_hat)
    except:
        traceback.print_exc()
        print("func: %s"%eval_func.__name__)
        print("params: "+ str(params ))
        print(h_hat)
    cdf_h = cdf(h[idx:])
    return max( [abs(cdf_h[i] - cdf_h_hat[i]) for i in range(len(cdf_h_hat))] )

def KSmod(params, eval_func, bins, bmin, h):
    idx = bins.index(bmin)
    h_hat = [eval_func(params, b) for b in bins[idx:len(h)]]
    return max( [abs(h_hat[i] - h[i+idx]) for i in range(len(h_hat))])

def estimate(h, bins, bmin, eval_func, obj_func=KSmod):
    params_guess = [1, 1, 1]
    res = minimize(obj_func, params_guess, args=(eval_func, bins, bmin, h), 
                   method='Nelder-Mead', options={'disp':False})
    return res.x

def estimate_bmin(h, bins, bmax=None, obj_func=KSmod, eval_funcs=[power]):
    results = None
    data = [[]]
    #verbose("Using function %s"%obj_func.__name__)
    verbose(h)
    minKS = 1
    if not bmax:
        bmax = bins[-1]
    bmin = bmax
    for bmin_test in bins[0:-2]:
        i = bins.index(bmin_test)
        data.append([])
        bminKS = minKS
        func_data = collections.OrderedDict()
        for eval_func in eval_funcs:
            j = eval_funcs.index(eval_func)
            data[i].append(None)
            params = estimate(h, bins, bmin_test, eval_func, obj_func)
            
            #compute KS statistic
            KS_stat = KS(params, eval_func, bins, bmin_test, h)
            if bmin_test<=bmax and KS_stat<bminKS:
                bminKS=KS_stat
            data[i][j] = (eval_func.__name__, params, KS_stat)
        if bmin_test<=bmax and bminKS<minKS:
            minKS = bminKS
            bmin = bmin_test
    return (bmin, data)

def generateData(h, bins, bmin, cdfs):
    idx = bins.index(bmin)
    N = sum(h)
    p = sum(h[idx:])/N
    data = {}
    for eval_func in cdfs.keys():
        data[eval_func] = [0 for count in h]
    
    for i in range(N):
        if random()>p:
            bin_num = randint(0, idx-1)
            for eval_func in cdfs.keys():
                data[eval_func][bin_num] += 1
        else:
            r = random()
            for eval_func in cdfs.keys():
                data[eval_func][fromCDF(r, cdfs[eval_func])] += 1
    return data

def goodness_of_fit(h, bins, bmin, eval_funcs, params, eps, obj_func=KSmod):
    '''
    eval_funcs : a list of evaluation functions
    params : a dictionary of estimated parameter vectors for the functions in eval_funcs
    '''
    counts = collections.OrderedDict()  #count of synthetic data sets with KS_synthetic > KS_est
    KS_est = {}
    num_data_sets = int((eps**-2)/4)+1
    verbose("Generating %d data sets..."%num_data_sets)
    idx = bins.index(bmin)
    # generate distributions from parameters
    distributions = {}
    for func in eval_funcs:
        distribution = []
        for i in range(idx):
            distribution.append(0)
        for i in range(idx, len(h)):
            distribution.append(func(params[func], bins[i]))
        print("%s distribution: "%func.__name__)
        print(distribution)
        distributions[func] = cdf(distribution)
        KS_est[func] = KS(params[func], func, bins, bmin, h)
        counts[func] = 0
    
    for i in range(num_data_sets):
        h_syn = generateData(h, bins, bmin, distributions)
        for func in eval_funcs:
            params_syn = estimate(h_syn[func], bins, bmin, func, obj_func=obj_func)
            KS_syn = KS(params_syn, func, bins, bmin, h_syn[func])            
            counts[func] += 1 if KS_syn>KS_est[func] else 0
    
    for func in eval_funcs:
        counts[func] /= num_data_sets
    return counts # the p-values for each of the estimated functions

def output_estimates(data, bins, funcs):
    verbose("bmin \t function \t param0 \t param1 \t KS")
    for func in funcs:
        for bmin in bins[0:-2]:        
            i = bins.index(bmin)
            j = funcs.index(func)
            func_name, params, KS_stat = data[i][j]
            verbose("%d, %s, %1.4f, %1.4f, %1.4f, %1.4f"%
                    (bmin, func_name, params[0], params[1], params[2], KS_stat))
    

if __name__ == '__main__':
    H = collections.OrderedDict()
    H["all"] = [301,199,86,74,59,59,26,7,18]
    H["LOTRO"] = [64,45,20,19,12,13,5,0,7]
    H["EQ2"] = [28,20,8,7,7,4,3,1,1]
    H["EVE"] = [41,29,12,11,8,4,2,1,1]
    H["AoC"] = [51,42,27,21,18,25,13,7,5]
    H["GW2"] = [117,63,19,16,14,13,3,2,8]
    bins = [1, 20, 50, 75, 100, 150, 300, 500, 1000, 1001]
    H["linear"] = [int(96.6481-0.1117*b) for b in bins[0:-1]]
    H["lognormal"] = [0, 212, 85, 57, 43, 28, 14, 9, 4]
    results = {}
    eval_funcs = [power, linear, exponential, lognormal]
    error_function = error_sumsq
    for game in H.keys():
        verbose("Game %s Analysis"%game)
        result = estimate_bmin(H[game], bins, bmax=300, obj_func=error_function, eval_funcs=eval_funcs)
        results[game] = result
        bmin, data = result
        output_estimates(data, bins, eval_funcs)
        verbose("bmin selected = %d"%bmin)
        verbose("--------------------")
        eps = 0.01
        i = bins.index(bmin)
        params = collections.OrderedDict()
        for func in eval_funcs:
            j = eval_funcs.index(func)
            fname, params[func], ks = data[i][j]
        p_values = goodness_of_fit(H[game], bins, bmin, eval_funcs, 
                                  params, eps, obj_func=error_function)
        verbose("p-values : ")
        for func in eval_funcs:
            verbose("%s , %1.4f"%(func.__name__, p_values[func]))
        verbose("****************************************************")
#     
    