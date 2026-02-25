

def one_step_EC(action, env, sparse_coding, pmap):
    s_e, reward_vector = env.step_EC(action)
    s_h = sparse_coding.one_step(s_e)
    p_h = pmap.one_step_from_sc(s_h)
    return s_e, reward_vector, s_h, p_h

def one_step_grid(action, env, sparse_coding, pmap):
    s_e, reward_vector = env.step_grid(action)
    s_h = sparse_coding.one_step(s_e)
    p_h = pmap.one_step_from_sc(s_h)
    return s_e, reward_vector, s_h, p_h

def one_step_grid_border(action, env, sparse_coding, pmap = None):
    s_e, reward_vector = env.step_grid_border(action)
    s_h = sparse_coding.one_step(s_e)
    if pmap is None:
        return s_e, reward_vector, s_h
    else:
        p_h = pmap.one_step_from_sc(s_h)
        return s_e, reward_vector, s_h, p_h
