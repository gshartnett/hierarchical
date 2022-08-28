from itertools import product
from collections import Counter
import functools
import numpy as onp #o for "old" or "original"
import jax.numpy as np
import jax
from jax import grad, jit, lax, random, ops, vmap, jacfwd, jacrev, device_get, device_put
from jax.lib import xla_bridge
print("jax version {}".format(jax.__version__))
print("jax backend {}".format(xla_bridge.get_backend().platform))


@functools.lru_cache(maxsize=None)
def parity_uniform(n, J, beta):
    '''The recursion relation for the uniform model parity.'''
    if n == 0:
        return onp.tanh(beta*J)
    else:
        return (onp.tanh(beta*J) + parity_uniform(n-1, J, beta)**2) / (1 + parity_uniform(n-1, J, beta)**2 * onp.tanh(beta*J)) 
    
    
def coth(x): 
    '''Hyperbolic cotangent'''
    return onp.cosh(x)/onp.sinh(x) 


def parity_uniform_analytic(J, beta):
    '''The analytic prediction for the parity of the uniform model.'''
    bJ = beta*J
    if bJ <= onp.log(2)/2:
        return 0.5*( (coth(bJ) - 1) - np.sign(bJ)*np.sqrt(coth(bJ)**2 - 2*coth(bJ) -3) )
    else:
        return 1
    
    
def compute_states(k):
    ''' 
    Generate all 2^(2^k) states at level k.
    Represent the states as binary bit strings, i.e., '01001100'
    '''
    N = 2**k
    return ["".join(str(bit) for bit in bits) for bits in product([0, 1], repeat=N)]


def compute_parity(state):
    '''Return the parity of the state.'''
    if len(state) > 1:
        return -(2*(state.count('1') % 2) - 1)
    elif state == '0':
        return -1
    else:
        return 1


def compute_uniform_Hamiltonian(state, J):
    '''Compute the energy of a state for the uniform model.'''
    N = len(state)
    if N > 1:
        stateL = state[:N//2]
        stateR = state[N//2:]
        HL = compute_uniform_Hamiltonian(stateL, J)
        HR = compute_uniform_Hamiltonian(stateR, J)
        return HL + HR - J*compute_parity(state)
    elif state == '0':
        return J
    else:
        return -J
    

def compute_uniform_gs_degen(k, J, parity=None):
    '''
    Compute the ground state energy and degenerarcy.
    Allows for filtering of states by parity, in which case the function 
    only returns the lowest energy state, which might not be the ground state.
    '''
    states = compute_states(k)
    if parity is not None:
        states = [s for s in states if compute_parity(s)==parity]
    state_dic = {state:compute_uniform_Hamiltonian(state, J) for state in states}
    counts = Counter(state_dic.values())
    Egs = min(counts.keys())
    dgs = counts[Egs]
    return Egs, dgs


def splitList(array):
    '''Given an array, split it into two halves.'''
    n = len(array)
    half = int(n/2) # py3
    return array[:half], array[n-half:]


def splitState(state):
    '''Given a string, split it into two halves.'''
    n = len(state)
    half = int(n/2)
    return state[:half], state[n-half:]


def Hamiltonian(state, coupling):
    '''Compute the energy of a state for a given list of couplings.'''
    N = len(state)
    assert len(coupling) == 2*N - 1
    if N == 1:
        return -coupling[0]*compute_parity(state)
    else:
        couplingL, couplingR = splitList(coupling[1:])
        stateL, stateR = splitState(state)
        return -coupling[0]*compute_parity(state) + Hamiltonian(stateL, couplingL) + Hamiltonian(stateR, couplingR)
    
    
def compute_spectrum(couplings):
    '''Given a list of couplings, compute the full spectrum.'''
    N = (len(couplings) + 1)//2
    n = int(np.log(N)/np.log(2))
    states = compute_states(n)
    return Counter([Hamiltonian(s, couplings) for s in states])


def degeneracy(n):
    '''
    Implements the recursion relations for the gs degeneracies of the 
    anti-ferromagnetic uniform model for levels k = 0, ..., n.
    '''
    dplus = {0: 1.0}
    dminus = {0: 1.0}
    dfull = {0: 1.0}
    for k in range(1,n+1):
        dminus[k] = 2 * dminus[k-1] * dplus[k-1]
        if k % 2 == 0:
            dplus[k] = dminus[k-1]**2 + dplus[k-1]**2
            dfull[k] = dminus[k] 
        else:
            dplus[k] = dminus[k-1]**2
            dfull[k] = dminus[k] + dplus[k]
    return {'minus':dminus, 'plus':dplus, 'full':dfull}


def log_degeneracy(n):
    '''
    Implements the recursion relations for the gs degeneracies of the 
    anti-ferromagnetic uniform model for levels k = 0, ..., n.
    '''
    gplus = {0: 0.0}
    gminus = {0: 0.0}
    gfull = {0: 0.0}
    for k in range(1,n+1):
        gminus[k] = 1 + gminus[k-1] + gplus[k-1]
        if k % 2 == 0:
            gplus[k] = 2*gminus[k-1] + onp.log(1 + 2**(2*(gplus[k-1]-gminus[k-1])))/onp.log(2)
            gfull[k] = gminus[k] 
        else:
            gplus[k] = 2*gminus[k-1]
            gfull[k] = gminus[k] + onp.log(1 + 2**(gplus[k]-gminus[k]))/onp.log(2)
    return {'minus':gminus, 'plus':gplus, 'full':gfull}


def GS_by_parity(couplings):
    '''
    This function recursively computes the lowest energy state, energy, 
    and degeneracy for each parity (+/-) superselection sector of the 
    state space. If the degeneracy is greater than 1, than the state
    returned is just one representative of the states counted by the 
    degeneracy.
    '''
    N = (len(couplings) + 1)//2
    k = int(onp.log(N)/onp.log(2))
    
    ## the leaf nodes
    if k == 0:
        gs_minus = {'s':onp.asarray([-1]), 'E':couplings[0], 'd':1, 'logd':0}
        gs_plus = {'s':onp.asarray([1]), 'E':-couplings[0], 'd':1, 'logd':0}
        return gs_minus, gs_plus    
    
    ## all other nodes
    else:
        couplingsL, couplingsR = splitList(couplings[1:])
        gs_minusL, gs_plusL = GS_by_parity(couplingsL)
        gs_minusR, gs_plusR = GS_by_parity(couplingsR)
        
        ## the lowest energy plus states are always of the form -- (why?)
        gs_minusminus = {'s':onp.concatenate((gs_minusL['s'], gs_minusR['s'])),
                         'E':gs_minusL['E'] + gs_minusR['E'] - couplings[0],
                         'd':gs_minusL['d'] * gs_minusR['d'],
                         'logd':gs_minusL['logd'] + gs_minusR['logd']
                         }        
        gs_plusplus = {'s':onp.concatenate((gs_plusL['s'], gs_plusR['s'])),
                       'E':gs_plusL['E'] + gs_plusR['E'] - couplings[0],
                       'd':gs_plusL['d'] * gs_plusR['d'],
                       'logd':gs_plusL['logd'] + gs_plusR['logd']
                       }   
        
        gs_minusplus = {'s':onp.concatenate((gs_minusL['s'], gs_plusR['s'])),
                         'E':gs_minusL['E'] + gs_plusR['E'] + couplings[0],
                         'd':gs_minusL['d'] * gs_plusR['d'],
                         'logd':gs_minusL['logd'] + gs_plusR['logd']
                         }
        gs_plusminus = {'s':onp.concatenate((gs_plusL['s'], gs_minusR['s'])),
                        'E':gs_plusL['E'] + gs_minusR['E'] + couplings[0],
                        'd':gs_plusL['d'] * gs_minusR['d'],
                        'logd':gs_plusL['logd'] + gs_minusR['logd']
                        }
        
        ## - states (+-, -+)
        if gs_minusplus['E'] < gs_plusminus['E']:
            gs_minus = gs_minusplus
        elif gs_plusminus['E'] < gs_minusplus['E']:
            gs_minus = gs_plusminus
        else:
            gs_minus = {'s':gs_minusplus['s'], 
                        'E':gs_minusplus['E'], 
                        'd':gs_minusplus['d'] + gs_plusminus['d'],
                        'logd':gs_minusplus['logd'] + onp.log(1 + 2.0**(gs_plusminus['logd'] - gs_minusplus['logd']))/onp.log(2)}

        ## + states (++, --)
        if gs_plusplus['E'] < gs_minusminus['E']:
            gs_plus = gs_plusplus
        elif gs_minusminus['E'] < gs_plusplus['E']:
            gs_plus = gs_minusminus
        else:
            gs_plus = {'s':gs_minusminus['s'], 
                       'E':gs_minusminus['E'], 
                       'd':gs_minusminus['d'] + gs_plusplus['d'],
                       'logd':gs_minusminus['logd'] + onp.log(1 + 2.0**(gs_plusplus['logd'] - gs_minusminus['logd']))/onp.log(2)}

        return gs_minus, gs_plus           
    
    
def GS(couplings):
    gs_minus, gs_plus = GS_by_parity(couplings)
    
    if gs_minus['E'] < gs_plus['E']:
        return gs_minus
    elif gs_plus['E'] < gs_minus['E']:
        return gs_plus
    else:
        return {'s':gs_minus['s'], 
                'E':gs_minus['E'], 
                'd':gs_minus['d'] + gs_plus['d'],
                'logd':gs_minus['logd'] + onp.log(1 + 2.0**(gs_plus['logd'] - gs_minus['logd']))/onp.log(2)} 


def fractional_GS_degen_by_parity(couplings):
    '''
    Compute the fractional ground state degeneracy by parity superselection sector.
    Returns a 2D array x, with x[0] the fraction of negative parity ground states, 
    and x[1] the fraction of positive parity ground states.
    '''
    gs_minus, gs_plus = GS_by_parity(couplings)
    if gs_minus['E'] < gs_plus['E']:
        return onp.asarray([1,0])
    elif gs_plus['E'] < gs_minus['E']:
        return onp.asarray([0,1])
    else:
        return onp.asarray([gs_minus['d']/(gs_minus['d'] + gs_plus['d']), gs_plus['d']/(gs_minus['d'] + gs_plus['d'])])
    
    
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


def tree_traversal(k, p=1, history=None):
    '''
    Traverses a balanced binary tree with height k, corresponding
    2*N-1 nodes, with N=2^k. 
    The traversal is done in pre-order (root, left, right), and 
    the function returns the list of nodes in the order visited.
    This is useful in ensuring that lists of coupling values are 
    sorted according to the pre-order standard.
    '''
    if history is None:
        history = []
    if k >= 0:
        history.append((k,p))
        tree_traversal(k-1, 2*p-1, history)
        tree_traversal(k-1, 2*p, history)
        return history
    
    
def coupling_dic_to_array(coupling_dic):
    '''
    Receives as input a dic of coupling values for
    each node in the binary tree, and returns an 
    array of coupling values, sorted according to the 
    pre-order binary tree traversal order.
    '''
    N = len(coupling_dic)
    n = int(onp.log(N)/onp.log(2))
    return onp.asarray([coupling_dic[node] for i, node in enumerate(tree_traversal(n))])


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
def heat_capacity(beta, couplings):
    '''
    Compute the heat capacity:
    C = - T \partial_T^2 F, or 
    C = \beta^2 \partial_{\beta}^2 \ln Z
    '''
    #return - beta**3*d2lnZ_by_dbeta(beta, couplings) - 2*beta**2*dlnZ_by_dbeta(beta, couplings)
    return beta**2 * d2lnZ_by_dbeta(beta, couplings)


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
    \frac{\beta^2}{N} \sum_{ij} (<s_i s_j> - <s_i><s_j>)^2.
    The gradient is taken wrt all the couplings, but we only need
    the first 2**n couplings (the local magnetic fields J_{0,p}).
    Note: this needs an expectation value over the disorder
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