def viterbi(obs, states, start_p, trans_p, emit_p):

    return viterbi_path(start_p, trans_p, obs, states)


import numpy as np


def viterbi_path(prior, transmat, obslik, states, scaled=True, ret_loglik=False):
    '''Finds the most-probable (Viterbi) path through the HMM state trellis
    Notation:
        Z[t] := Observation at time t
        Q[t] := Hidden state at time t
    Inputs:
        prior: np.array(num_hid)
            prior[i] := Pr(Q[0] == i)
        transmat: np.ndarray((num_hid,num_hid))
            transmat[i,j] := Pr(Q[t+1] == j | Q[t] == i)
        obslik: np.ndarray((num_hid,num_obs))
            obslik[i,t] := Pr(Z[t] | Q[t] == i)
        scaled: bool
            whether or not to normalize the probability trellis along the way
            doing so prevents underflow by repeated multiplications of probabilities
        ret_loglik: bool
            whether or not to return the log-likelihood of the best path
    Outputs:
        path: np.array(num_obs)
            path[t] := Q[t]
    '''
    num_hid = len(states) #obslik.shape[0] # number of hidden states
    num_obs = len(obslik)# obslik.shape[1] # number of observations (not observation *states*)

    # trellis_prob[i,t] := Pr((best sequence of length t-1 goes to state i), Z[1:(t+1)])
    trellis_prob = np.zeros((num_hid,num_obs))
    # trellis_state[i,t] := best predecessor state given that we ended up in state i at t
    trellis_state = np.zeros((num_hid,num_obs), dtype=int) # int because its elements will be used as indicies
    path = np.zeros(num_obs, dtype=int) # int because its elements will be used as indicies

    #trellis_prob[:,0] = prior * obslik[:,0] # element-wise mult

    #print("numhid {}".format(num_hid))
    for pi in range(num_hid):
        trellis_prob[pi, 0] = prior[states[pi]] * obslik[0][states[pi]]


    if scaled:
        scale = np.ones(num_obs) # only instantiated if necessary to save memory
        scale[0] = 1.0 / np.sum(trellis_prob[:,0])
        trellis_prob[:,0] *= scale[0]

    trellis_state[:,0] = 0 # arbitrary value since t == 0 has no predecessor
    for t in range(1, num_obs):
        for j in range(num_hid):
            #trans_probs = trellis_prob[:,t-1] * transmat[:,j] # element-wise mult

            trans_probs = []
            for tpi in range(num_hid):
                trans_probs.append(trellis_prob[tpi,t-1] * transmat.get(states[tpi] + "_" + states[j], 0.0001))

            trans_probs = np.array(trans_probs)

            trellis_state[j,t] = trans_probs.argmax()
            trellis_prob[j,t] = trans_probs[trellis_state[j,t]] # max of trans_probs
            trellis_prob[j,t] *= obslik[t][states[j]] #[j,t]

        if scaled:
            scale[t] = 1.0 / np.sum(trellis_prob[:,t])
            trellis_prob[:,t] *= scale[t]

    path[-1] = trellis_prob[:,-1].argmax()
    for t in range(num_obs-2, -1, -1):
        path[t] = trellis_state[(path[t+1]), t+1]

    if not ret_loglik:
        return [states[p] for p in path]
    else:
        if scaled:
            loglik = -np.sum(np.log(scale))
        else:
            p = trellis_prob[path[-1],-1]
            loglik = np.log(p)
        return path, loglik

def viterbiold(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p(st, obs[0]), "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t - 1][states[0]]["prob"] *  trans_p.get(states[0] + "_" + st, 0.0001)  #trans_p[states[0] + "_" + st]# trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p.get(states[0] + "_" + st, 0.0001) #trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p(st, obs[t])
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

    for line in dptable(V):
        #print(line)
        pass
    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    #print('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)

    return opt


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)