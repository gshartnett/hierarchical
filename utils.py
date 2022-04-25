import os
import numpy as onp #o for "old" or "original"
import jax.numpy as np
import jax
from jax import grad, jit, lax, random, ops, vmap, jacfwd, jacrev, device_get, device_put
from jax.lib import xla_bridge
print("jax version {}".format(jax.__version__))
print("jax backend {}".format(xla_bridge.get_backend().platform))


def generate_couplings_dic(n, prob=0.5, sigma=0):
    '''
    Generate a realization of the couplings:
    J_{k,p} = 2^{k \sigma} x, where x is {-1,1} with 
    probability {1-prob, prob}. Returns a dict.
    '''
    couplings = {(k,p): (2**(k*sigma))*(2*onp.random.binomial(1, prob) - 1) for k in range(n+1) for p in range(1,2**(n-k)+1)}
    return couplings


def generate_couplings(n, prob=0.5, sigma=0):
    '''
    Generate a realization of the couplings:
    J_{k,p} = 2^{k \sigma} x, where x is {-1,1} with 
    probability {1-prob, prob}. Returns an array.
    '''
    couplings = {(k,p): (2**(k*sigma))*(2*onp.random.binomial(1, prob) - 1.0) for k in range(n+1) for p in range(1,2**(n-k)+1)}
    return np.asarray(list(couplings.values()))


def find_parities_dic(beta, couplings):
    '''
    Using the recursion relation, compute the parities at 
    all levels in the hierarchical tree.
    '''

    n = int(onp.log((len(couplings)+1)/2)/onp.log(2))  
    parities = {(k,p):0 for k in range(n+1) for p in range(1,2**(n-k)+1)}

    for k in range(0, n+1):
        for p in range(1,2**(n-k)+1):

            betaJ = beta*couplings[k,p]

            if k == 0:
                parities[k,p] = onp.tanh(betaJ)
            else:
                parities[k,p] = (onp.sinh(betaJ) + onp.cosh(betaJ)*parities[k-1,2*p-1]*parities[k-1,2*p])
                parities[k,p] /= (onp.cosh(betaJ) + onp.sinh(betaJ)*parities[k-1,2*p-1]*parities[k-1,2*p])
                
    return parities


def Nishimorhi_beta(prob):
    '''
    Compute the Nishimori inverse temperature,
    \beta = \frac{1}{2} \ln\left( \frac{p}{1-p})
    '''
    return 0.5*onp.log(prob/(1-prob))


def index_dictionaries(n):
    '''
    Find the map and inverse map from the tree indices (k,p)
    to a 1d array index i=1,...
    '''
    i = 0
    i_to_kp = {}
    kp_to_i = {}
    for k in range(0, n+1):
        for p in range(1, 2**(n-k)+1):
            i_to_kp[i] = (k,p)
            kp_to_i[(k,p)] = i
            i += 1
            
    return i_to_kp, kp_to_i


@jit
def find_parities(beta, couplings):
    '''
    Using the recursion relation, compute the parities at 
    all levels in the hierarchical tree.
    Note: this is rather slow.
    '''
    
    n = int(onp.log((len(couplings)+1)/2)/onp.log(2)) 
    i_to_kp, kp_to_i = index_dictionaries(n)    
    parities = np.zeros(len(couplings))
    
    for k in range(0, n+1):
        for p in range(1,2**(n-k)+1):
            i = kp_to_i[(k,p)]
            
            betaJ = beta*couplings[i]

            if k == 0:
                new_value = np.tanh(betaJ)
                #parities = ops.index_update(parities, i, new_value)
                parities = parities.at[i].set(new_value)
            else:
                i_left = kp_to_i[(k-1,2*p-1)]
                i_right = kp_to_i[(k-1,2*p)]                
                new_value = (np.sinh(betaJ) + np.cosh(betaJ)*parities[i_left]*parities[i_right])
                new_value /= (np.cosh(betaJ) + np.sinh(betaJ)*parities[i_left]*parities[i_right])
                #parities = ops.index_update(parities, i, new_value)
                parities = parities.at[i].set(new_value)
                
    return parities


@jit
def log_partition_function(beta, couplings):
    '''
    Using the recursion relation, compute the log partition function.
    Note: this is rather slow.
    Note: I did a quick & dirty check of this against my Mathematica code.
    '''
    n = int(onp.log((len(couplings)+1)/2)/onp.log(2))
    i_to_kp, kp_to_i = index_dictionaries(n)    
    parities = find_parities(beta, couplings)    
    lnZ = np.zeros(len(parities))

    for k in range(0, n+1):
        for p in range(1,2**(n-k)+1):
            i = kp_to_i[(k,p)]
            betaJ = beta*couplings[i]

            if k == 0:
                new_value = np.log(2*np.cosh(betaJ))
                #lnZ = ops.index_update(lnZ, i, new_value)
                lnZ = lnZ.at[i].set(new_value)
            else:
                i_left = kp_to_i[(k-1,2*p-1)]
                i_right = kp_to_i[(k-1,2*p)]                
                new_value = lnZ[i_left] + lnZ[i_right] + np.log(np.cosh(betaJ) + parities[i_left]*parities[i_right]*np.sinh(betaJ))
                #lnZ = ops.index_update(lnZ, i, new_value)
                lnZ = lnZ.at[i].set(new_value)                
    return lnZ


@jit
def dlnZ_by_dJ(beta, couplings):
    '''
    Compute the gradient of the log partition function wrt to the couplings
    '''
    return grad(lambda x: log_partition_function(beta, x)[-1])(couplings)


@jit
def dlnZ_by_dbeta(beta, couplings):
    '''
    Compute the derivative of the log partition function wrt to inverse 
    temperature
    '''
    return grad(lambda x: log_partition_function(x, couplings)[-1])(beta)


@jit
def d2lnZ_by_dbeta(beta, couplings):
    '''
    Compute the second derivative of the log partition function wrt to 
    inverse temperature
    '''
    return grad(lambda x: dlnZ_by_dbeta(x, couplings))(beta)


@jit 
def free_energy(beta, couplings):
    '''Compute the free energy'''
    return - log_partition_function(beta, couplings)[-1]/beta
        

@jit
def energy(beta, couplings):
    '''Compute the energy'''
    return - dlnZ_by_dbeta(beta, couplings)


@jit
def specific_heat(beta, couplings):
    '''
    Compute the specific heat:
    C = - T \partial_T^2 F, or 
    C = (-\beta^3 \partial_{\beta}^2 - 2 \beta^2 \partial_{\beta} F)
    '''
    return - beta**3*d2lnZ_by_dbeta(beta, couplings) - 2*beta**2*d2lnZ_by_dbeta(beta, couplings)


@jit
def entropy(beta, couplings):
    '''Compute the entropy'''
    return beta*(energy(beta, couplings) - free_energy(beta, couplings))


@jit
def d2lnZ_by_dJ(beta, couplings):
    '''
    Compute the second derivative of the log partition function wrt to 
    the couplings J_{k,p}
    '''
    return jax.jacfwd(grad(lambda x: log_partition_function(beta, x)[-1]))(couplings)


@jit
def chi_SG(beta, couplings):
    '''
    The spin-glass susceptability
    \frac{\beta^2}{N} \sum_{ij} (<s_i s_j> - <s_i><s_j>)^2
    Note: this need an expectation value over the disorder
    '''
    n = int(onp.log((len(couplings)+1)/2)/onp.log(2))
    return np.sum(np.square(d2lnZ_by_dJ(beta, couplings)[0:2**n, 0:2**n]))/2**n


@jit
def chi_ij(beta, couplings):
    '''
    The spin-glass susceptability matrix
    (<s_i s_j> - <s_i><s_j>)
    '''
    n = int(onp.log((len(couplings)+1)/2)/onp.log(2))
    return d2lnZ_by_dJ(beta, couplings)[0:2**n, 0:2**n]/beta**2