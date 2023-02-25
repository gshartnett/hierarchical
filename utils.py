from itertools import product
from collections import Counter
import functools
import numpy as onp #o for "old" or "original"
import jax.numpy as np
import jax
from jax import grad, jit
from jax.lib import xla_bridge
print("jax version {}".format(jax.__version__))
print("jax backend {}".format(xla_bridge.get_backend().platform))


@functools.lru_cache(maxsize=None)
def parity_uniform(n, J, beta):
    """
    The recursion relation for the uniform model parity.

    Parameters
    ----------
    n : int
        The log (base 2) of the number of spins.
    J : float
        The spin coupling.
    beta : float
        Inverse temperature.

    Returns
    -------
    float
        The parity expectation value <S_i>, which is independent of i
        in the uniform model.
    """
    if n == 0:
        return onp.tanh(beta*J)
    return (
        (onp.tanh(beta*J) + parity_uniform(n-1, J, beta)**2) /
        (1 + parity_uniform(n-1, J, beta)**2 * onp.tanh(beta*J))
    )


def coth(x):
    """
    Hyperbolic cotangent

    Parameters
    ----------
    x : float
        Argument.

    Returns
    -------
    float
        Output.
    """
    return onp.cosh(x)/onp.sinh(x)


def parity_uniform_analytic(J, beta):
    """
    The analytic prediction for the parity of the uniform model in the
    infinite n limit.

    Parameters
    ----------
    J : float
        The spin coupling.
    beta : float
        Inverse temperature.

    Returns
    -------
    float
        The parity expectation value <S_i>, which is independent of i
        in the uniform model.
    """
    beta_J = beta*J
    if beta_J <= onp.log(2)/2:
        return 0.5*(
                    (coth(beta_J) - 1) - np.sign(beta_J) *
                    np.sqrt(coth(beta_J)**2 - 2*coth(beta_J) -3)
                    )
    else:
        return 1


def compute_states(n):
    """
    Generate all 2^(2^n) states at level n.
    Represent the states as binary bit strings, i.e., '01001100'.

    Parameters
    ----------
    n : int
        The log (base 2) of the number of spins.

    Returns
    -------
    List
        The list of all system states, represented as bitstrings.
    """
    N = 2**n
    return ["".join(str(bit) for bit in bits) for bits in product([0, 1], repeat=N)]


def compute_parity(state):
    """
    Return the parity of the state.

    Parameters
    ----------
    state : str
        The state bitstring.

    Returns
    -------
    int
        The parity (+/- 1).
    """
    if len(state) > 1:
        return -(2*(state.count('1') % 2) - 1)
    if state == '0':
        return -1
    return 1


def compute_energy_uniform(state, J):
    """
    Compute the energy of a state for the uniform model.

    Parameters
    ----------
    state : str
        The state bitstring.
    J : float
        The spin coupling.

    Returns
    -------
    float
        The energy of a given state.
    """
    N = len(state)
    if N > 1:
        state_left = state[:N//2]
        state_right = state[N//2:]
        energy_left = compute_energy_uniform(state_left, J)
        energy_right = compute_energy_uniform(state_right, J)
        return energy_left + energy_right - J*compute_parity(state)
    if state == '0':
        return J
    return -J


def compute_uniform_gs_degen(n, J, parity=None):
    """
    Compute the ground state energy and degenerarcy.
    Allows for filtering of states by parity, in which case the function
    only returns the lowest energy state, which might not be the ground state.

    Parameters
    ----------
    n : int
        The log (base 2) of the number of spins.
    J : float
        The spin coupling.
    parity : int, optional
        The parity, could be -1, +1, or None. By default None

    Returns
    -------
    Tuple
        The ground state energy and degeneracy.
    """
    states = compute_states(n)
    if parity is not None:
        assert parity in [-1, 1]
        states = [s for s in states if compute_parity(s)==parity]
    state_dic = {state:compute_energy_uniform(state, J) for state in states}
    counts = Counter(state_dic.values())
    energy_gs = min(counts.keys())
    degen_gs = counts[energy_gs]
    return energy_gs, degen_gs


def split_list(array):
    """
    Given an array, split it into two halves.

    Parameters
    ----------
    array : List
        A list or numpy array.

    Returns
    -------
    List
        A list or numpy array.
    """
    len_array = len(array)
    len_half = int(len_array/2)
    return array[:len_half], array[len_array-len_half:]


def split_state(state):
    """
    Given a string, split it into two halves.

    Parameters
    ----------
    state : str
        The state bitstring.

    Returns
    -------
    Tuple[str, str]
        The left and right state bitstrings.
    """
    len_array = len(state)
    len_half = int(len_array/2)
    return state[:len_half], state[len_array-len_half:]


def compute_energy(state, coupling):
    """
    Compute the energy of a state for a given list of couplings.

    Parameters
    ----------
    state : str
        The state bitstring.
    coupling : List
        The list of couplings.

    Returns
    -------
    float
        The energy of the state.
    """
    assert len(coupling) == 2*len(state) - 1
    if len(state) == 1:
        return -coupling[0]*compute_parity(state)
    coupling_left, coupling_right = split_list(coupling[1:])
    state_left, state_right = split_state(state)
    return (
        -coupling[0]*compute_parity(state)
        + compute_energy(state_left, coupling_left)
        + compute_energy(state_right, coupling_right)
    )


def compute_spectrum(couplings):
    """
    Given a list of couplings, compute the full spectrum.

    Parameters
    ----------
    couplings : List
        A list of the couplings.

    Returns
    -------
    Dict
        A dictionary of the form {energy: degeneracy}
    """
    num_spins = (len(couplings) + 1)//2
    n = int(np.log(num_spins)/np.log(2))
    states = compute_states(n)
    return Counter([compute_energy(s, couplings) for s in states])


def degeneracy(n):
    """
    Implements the recursion relations for the gs degeneracies of the
    anti-ferromagnetic uniform model for levels k = 0, ..., n.

    Parameters
    ----------
    n : int
        The log (base 2) of the number of spins.

    Returns
    -------
    Dict
        A dictionary containing the 3 degeneracies
        (- parity states, + parity states, and full system)
    """
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
    """
    Implements the recursion relations for the gs degeneracies of the
    anti-ferromagnetic uniform model for levels k = 0, ..., n.
    In order to prevent numerical overflow, the log (base 2) degeneracies
    are computed, rather than the actual degeneracies.

    Parameters
    ----------
    n : int
        The log (base 2) of the number of spins.

    Returns
    -------
    Dict
        A dictionary containing the 3 log degeneracies
        (- parity states, + parity states, and full system)
    """
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


def compute_ground_state_parity(couplings):
    """
    This function recursively computes the lowest energy state, energy,
    and degeneracy for each parity (+/-) superselection sector of the
    state space. If the degeneracy is greater than 1, than the state
    returned is just one representative of the states counted by the
    degeneracy.

    Parameters
    ----------
    couplings : List
        The couplings.

    Returns
    -------
    Tuple
        A tuple containing the ground state dictionaries for the + and - sectors.
    """
    num_spins = (len(couplings)+1)//2
    n = int(onp.log(num_spins)/onp.log(2))

    ## the leaf nodes
    if n == 0:
        gs_minus = {'s':onp.asarray([-1]), 'E':couplings[0], 'd':1, 'logd':0}
        gs_plus = {'s':onp.asarray([1]), 'E':-couplings[0], 'd':1, 'logd':0}
        return gs_minus, gs_plus

    ## all other nodes
    else:
        couplings_left, couplings_right = split_list(couplings[1:])
        gs_minus_left, gs_plus_left = compute_ground_state_parity(couplings_left)
        gs_minus_right, gs_plus_right = compute_ground_state_parity(couplings_right)

        ## the lowest energy plus states are always of the form -- (why?)
        gs_minusminus = {'s':onp.concatenate((gs_minus_left['s'], gs_minus_right['s'])),
                         'E':gs_minus_left['E'] + gs_minus_right['E'] - couplings[0],
                         'd':gs_minus_left['d'] * gs_minus_right['d'],
                         'logd':gs_minus_left['logd'] + gs_minus_right['logd']
                         }
        gs_plusplus = {'s':onp.concatenate((gs_plus_left['s'], gs_plus_right['s'])),
                       'E':gs_plus_left['E'] + gs_plus_right['E'] - couplings[0],
                       'd':gs_plus_left['d'] * gs_plus_right['d'],
                       'logd':gs_plus_left['logd'] + gs_plus_right['logd']
                       }

        gs_minusplus = {'s':onp.concatenate((gs_minus_left['s'], gs_plus_right['s'])),
                         'E':gs_minus_left['E'] + gs_plus_right['E'] + couplings[0],
                         'd':gs_minus_left['d'] * gs_plus_right['d'],
                         'logd':gs_minus_left['logd'] + gs_plus_right['logd']
                         }
        gs_plusminus = {'s':onp.concatenate((gs_plus_left['s'], gs_minus_right['s'])),
                        'E':gs_plus_left['E'] + gs_minus_right['E'] + couplings[0],
                        'd':gs_plus_left['d'] * gs_minus_right['d'],
                        'logd':gs_plus_left['logd'] + gs_minus_right['logd']
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


def compute_ground_state(couplings):
    """
    This function recursively computes the lowest energy state, energy,
    and degeneracy. If the degeneracy is greater than 1, than the state
    returned is just one representative of the states counted by the
    degeneracy.

    Parameters
    ----------
    couplings : List
        The couplings.

    Returns
    -------
    Tuple
        The ground state dictionary.
    """
    ## compute the degeneracies by parity sector (+/-)
    gs_minus, gs_plus = compute_ground_state_parity(couplings)

    ## case 1: - has lower energy than +
    if gs_minus['E'] < gs_plus['E']:
        return gs_minus
    ## case 2: + has lower energy than -
    if gs_plus['E'] < gs_minus['E']:
        return gs_plus

    ## case 3: + and - have same energy
    return {
        's':gs_minus['s'],
        'E':gs_minus['E'],
        'd':gs_minus['d'] + gs_plus['d'],
        'logd':gs_minus['logd'] + onp.log(1 + 2.0**(gs_plus['logd'] - gs_minus['logd']))/onp.log(2)
        }


def fractional_ground_state_degen_by_parity(couplings):
    """
    Compute the fractional ground state degeneracy by parity superselection
    sector. Returns a 2D array x, with x[0] the fraction of negative parity
    ground states, and x[1] the fraction of positive parity ground states.

    Parameters
    ----------
    couplings : List
        The couplings.

    Returns
    -------
    List
        A list containing the fraction of ground states with each parity.
    """
    gs_minus, gs_plus = compute_ground_state_parity(couplings)
    if gs_minus['E'] < gs_plus['E']:
        return onp.asarray([1,0])
    if gs_plus['E'] < gs_minus['E']:
        return onp.asarray([0,1])
    return onp.asarray(
        [gs_minus['d']/(gs_minus['d'] + gs_plus['d']), gs_plus['d']/(gs_minus['d'] + gs_plus['d'])]
        )


def generate_couplings_dic(n, prob=0.5, sigma=0):
    """
    Generate a realization of the couplings:
    J_{k,p} = 2^{k \sigma} x, where x is {-1,1} with
    probability {1-prob, prob}. Returns a dict.

    Parameters
    ----------
    n : int
        The log (base 2) of the number of spins.
    prob : float, optional
        Disorder probability, by default 0.5
    sigma : int, optional
        Coupling scaling parameter, by default 0

    Returns
    -------
    Dict
        A dictionary of couplings. The keys are the coordinates of the
        nodes of the balanced binary tree.
    """
    couplings = {(k,p): (2**(k*sigma))*(2*onp.random.binomial(1, prob) - 1) for k in range(n+1) for p in range(1,2**(n-k)+1)}
    return couplings


def generate_couplings(n, prob=0.5, sigma=0):
    """
    Generate a realization of the couplings:
    J_{k,p} = 2^{k \sigma} x, where x is {-1,1} with
    probability {1-prob, prob}. Returns an array.

    Parameters
    ----------
    n : int
        The log (base 2) of the number of spins.
    prob : float, optional
        Disorder probability, by default 0.5
    sigma : int, optional
        Coupling scaling parameter, by default 0

    Returns
    -------
    List
        A list of couplings.
    """
    couplings = {(k,p): (2**(k*sigma))*(2*onp.random.binomial(1, prob) - 1.0) for k in range(n+1) for p in range(1,2**(n-k)+1)}
    return np.asarray(list(couplings.values()))


def tree_traversal(k, p=1, history=None):
    """
    Traverses a balanced binary tree with height k, corresponding
    2*N-1 nodes, with N=2^k.
    The traversal is done in pre-order (root, left, right), and
    the function returns the list of nodes in the order visited.
    This is useful in ensuring that lists of coupling values are
    sorted according to the pre-order standard.

    Parameters
    ----------
    k : int
        Height of balanced binary tree.
    p : int, optional
        The width coordinate, by default 1.
    history : List, optional
        A list of all the visited node coordiinates, by default None.

    Returns
    -------
    _type_
        _description_
    """
    if history is None:
        history = []
    if k >= 0:
        history.append((k,p))
        tree_traversal(k-1, 2*p-1, history)
        tree_traversal(k-1, 2*p, history)
        return history


def coupling_dic_to_array(coupling_dic):
    """
    Receives as input a dic of coupling values for
    each node in the binary tree, and returns an
    array of coupling values, sorted according to the
    pre-order binary tree traversal order.

    Parameters
    ----------
    coupling_dic : Dict
        Dictionary of couplings. The keys are the node coordinates.

    Returns
    -------
    Array
        Array of coupling values.
    """
    num_spins = (len(coupling_dic) + 1)//2
    n = int(onp.log(num_spins)/onp.log(2))
    return onp.asarray([coupling_dic[node] for i, node in enumerate(tree_traversal(n))])


def find_parities_dic(beta, couplings):
    """
    Using the recursion relation, compute the parities at
    all levels in the hierarchical tree.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    couplings : Array
        An array of couplings, with couplings[k,p] corresponding
        to the (k,p) node.

    Returns
    -------
    Array
        An dictionary of the parities, with parity[(k,p)] corresponding
        to the (k,p) node.
    """
    num_spins = (len(couplings)+1)//2
    n = int(onp.log(num_spins)/onp.log(2))
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
    """
    Compute the Nishimori inverse temperature,
    \beta = \frac{1}{2} \ln\left( \frac{p}{1-p})

    Parameters
    ----------
    prob : float
        The disorder probability, p.

    Returns
    -------
    float
        The inverse temperature.
    """
    assert (0 <= prob) and (prob <= 1)
    return 0.5*onp.log(prob/(1-prob))


def index_dictionaries(n):
    """
    Find the map and inverse map from the tree indices (k,p)
    to a 1d array index i=0,...,2N

    Parameters
    ----------
    n : int
        Log (base 2) of system size, i.e., N=2^n.

    Returns
    -------
    Tuple
        A tuple of dictionaries mapping b/w the traversal order of the
        nodes i=0,1,...,2N and the (k,p) coordinates.
    """
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
    """
    Using the recursion relation, compute the parities at
    all levels in the hierarchical tree.
    Note: this is rather slow

    Parameters
    ----------
    beta : float
        Inverse temperature.
    couplings : Array
        An array of couplings.

    Returns
    -------
    Array
        An array of parity values.
    """
    num_spins = (len(couplings)+1)//2
    n = int(onp.log(num_spins)/onp.log(2))
    _, kp_to_i = index_dictionaries(n)
    parities = np.zeros(len(couplings))

    ## loop over the (k,p) coordinates
    for k in range(0, n+1):
        for p in range(1,2**(n-k)+1):

            ## the traversal order index of the node (k,p)
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
    """
    Using the recursion relation, compute the log partition function.
    Note: this is rather slow.
    Note: I did a quick & dirty check of this against my Mathematica code.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    couplings : Array
        List of coupling values.

    Returns
    -------
    float
        Log paritition function.
    """
    num_spins = (len(couplings)+1)//2
    n = int(onp.log(num_spins)/onp.log(2))
    _, kp_to_i = index_dictionaries(n)
    parities = find_parities(beta, couplings)
    lnZ = np.zeros(len(parities))

    ## loop over the (k,p) node coordinates
    for k in range(0, n+1):
        for p in range(1, 2**(n-k)+1):

            i = kp_to_i[(k,p)]
            betaJ = beta*couplings[i]

            if k == 0:
                new_value = np.log(2*np.cosh(betaJ))
                #lnZ = ops.index_update(lnZ, i, new_value)
                lnZ = lnZ.at[i].set(new_value)
            else:
                i_left = kp_to_i[(k-1,2*p-1)]
                i_right = kp_to_i[(k-1,2*p)]
                new_value = (
                    lnZ[i_left] + lnZ[i_right]
                    + np.log(np.cosh(betaJ) + parities[i_left]*parities[i_right]*np.sinh(betaJ))
                )
                #lnZ = ops.index_update(lnZ, i, new_value)
                lnZ = lnZ.at[i].set(new_value)
    return lnZ


@jit
def dlnZ_by_dJ(beta, couplings):
    """
    Compute the gradient of the log partition function wrt to the couplings

    Parameters
    ----------
    beta : float
        Inverse temperature
    couplings : Array
        The couplings.

    Returns
    -------
    Array
        The gradient d(ln Z)/d(J).
    """
    return grad(lambda x: log_partition_function(beta, x)[-1])(couplings)


@jit
def dlnZ_by_dbeta(beta, couplings):
    """
    Compute the derivative of the log partition function wrt to inverse
    temperature.

    Parameters
    ----------
    beta : float
        Inverse temperature
    couplings : Array
        The couplings.

    Returns
    -------
    Array
        The gradient d(ln Z)/d(beta).
    """
    return grad(lambda x: log_partition_function(x, couplings)[-1])(beta)


@jit
def d2lnZ_by_dbeta(beta, couplings):
    """
    Compute the second derivative of the log partition function wrt to
    inverse temperature.

    Parameters
    ----------
    beta : float
        Inverse temperature
    couplings : Array
        The couplings.

    Returns
    -------
    Array
        The gradient d^2(ln Z)/d(beta)^2.
    """
    return grad(lambda x: dlnZ_by_dbeta(x, couplings))(beta)


@jit
def free_energy(beta, couplings):
    """
    Compute the free energy.

    Parameters
    ----------
    beta : float
        Inverse temperature
    couplings : Array
        The couplings.

    Returns
    -------
    float
        The free energy.
    """
    return - log_partition_function(beta, couplings)[-1]/beta


@jit
def energy(beta, couplings):
    """
    Compute the energy.

    Parameters
    ----------
    beta : float
        Inverse temperature
    couplings : Array
        The couplings.

    Returns
    -------
    float
        The energy.
    """
    return - dlnZ_by_dbeta(beta, couplings)


@jit
def heat_capacity(beta, couplings):
    """
    Compute the heat capacity:
    C = - T \partial_T^2 F, or
    C = \beta^2 \partial_{\beta}^2 \ln Z

    Parameters
    ----------
    beta : float
        Inverse temperature
    couplings : Array
        The couplings.

    Returns
    -------
    float
        The heat capacity
    """
    #return - beta**3*d2lnZ_by_dbeta(beta, couplings) - 2*beta**2*dlnZ_by_dbeta(beta, couplings)
    return beta**2 * d2lnZ_by_dbeta(beta, couplings)


@jit
def entropy(beta, couplings):
    """
    Compute the entropy.

    Parameters
    ----------
    beta : float
        Inverse temperature
    couplings : Array
        The couplings.

    Returns
    -------
    float
        The entropy.
    """
    return beta*(energy(beta, couplings) - free_energy(beta, couplings))


@jit
def d2lnZ_by_dJ(beta, couplings):
    """
    Compute the second derivative of the log partition function wrt to
    the couplings J_{k,p}.

    Parameters
    ----------
    beta : float
        Inverse temperature
    couplings : Array
        The couplings.

    Returns
    -------
    Array
        The matrix of second derivatives.
    """
    return jax.jacfwd(grad(lambda x: log_partition_function(beta, x)[-1]))(couplings)


@jit
def chi_SG(beta, couplings):
    """
    The spin-glass susceptability
    \frac{\beta^2}{N} \sum_{ij} (<s_i s_j> - <s_i><s_j>)^2.
    The gradient is taken wrt all the couplings, but we only need
    the first 2**n couplings (the local magnetic fields J_{0,p}).
    Note: this needs an expectation value over the disorder.

    Parameters
    ----------
    beta : float
        Inverse temperature
    couplings : Array
        The couplings.

    Returns
    -------
    float
        The spin-glass susceptability.
    """
    num_spins = (len(couplings)+1)//2
    n = int(onp.log(num_spins)/onp.log(2))
    return np.sum(np.square(d2lnZ_by_dJ(beta, couplings)[0:2**n, 0:2**n]))/2**n


@jit
def chi_ij(beta, couplings):
    """
    The spin-glass susceptability matrix
    (<s_i s_j> - <s_i><s_j>).

    Parameters
    ----------
    beta : float
        Inverse temperature
    couplings : Array
        The couplings.

    Returns
    -------
    Array
        The spin-glass susceptability matrix.
    """
    num_spins = (len(couplings)+1)//2
    n = int(onp.log(num_spins)/onp.log(2))
    return d2lnZ_by_dJ(beta, couplings)[0:2**n, 0:2**n]/beta**2