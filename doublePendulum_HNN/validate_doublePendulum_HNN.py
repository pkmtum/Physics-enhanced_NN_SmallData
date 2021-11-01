import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.ode import odeint
from functools import partial
# import matplotlib.pyplot as plt
import pickle
import os


def wrap_state(state):
    # wrap generalized coordinates to [-pi, pi)
    phi1 = (state[0] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    phi2 = (state[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    phi1_t = state[2]
    phi2_t = state[3]
    return jnp.array([phi1, phi2, phi1_t, phi2_t])


def analytical_dynamics(can_state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.81):
    """Analytical time derivative of the double pendulums state in
    generalized coordinates and conjugated momenta=(phi1, phi2, p1, p2)"""
    phi1, phi2, p1, p2 = can_state

    C0 = l1 * l2 * (m1 + m2 * jnp.sin(phi1 - phi2) ** 2)
    C1 = (p1 * p2 * jnp.sin(phi1 - phi2)) / C0
    C2 = (m2 * (l2 * p1) ** 2 + (m1 + m2) * (l1 * p2) ** 2 -
          2 * l1 * l2 * m2 * p1 * p2 * jnp.cos(phi1 - phi2)) * \
         jnp.sin(2 * (phi1 - phi2)) / (2 * C0 ** 2)

    # F is the right-hand side of the Hamilton's equations
    phi1_t = (l2 * p1 - l1 * p2 * jnp.cos(phi1 - phi2)) / (l1 * C0)
    phi2_t = (l1 * (m1 + m2) * p2 - l2 *
            m2 * p1 * jnp.cos(phi1 - phi2)) / (l2 * m2 * C0)
    p1_t = -(m1 + m2) * g * l1 * jnp.sin(phi1) - C1 + C2
    p2_t = -m2 * g * l2 * jnp.sin(phi2) + C1 - C2

    can_state_t = jnp.array([phi1_t, phi2_t, p1_t, p2_t])
    return can_state_t


# set phi=0 position to minimal potential energy of 9.81*m1*l2
def analytical_hamiltonian(can_state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.81):
    phi1, phi2, p1, p2 = can_state
    C0 = l1 * l2 * (m1 + m2 * jnp.sin(phi1 - phi2) ** 2)
    w1 = (l2 * p1 - l1 * p2 * jnp.cos(phi1 - phi2)) / (l1 * C0)
    w2 = (l1 * (m1 + m2) * p2 - l2 *
          m2 * p1 * jnp.cos(phi1 - phi2)) / (l2 * m2 * C0)
    # compute the kinetic energy of each bob
    K1 = 0.5 * m1 * (l1 * w1) ** 2
    K2 = 0.5 * m2 * ((l1 * w1) ** 2 + (l2 * w2) ** 2 +
                     2 * l1 * l2 * w1 * w2 * jnp.cos(phi1 - phi2))
    T = K1 + K2
    # compute the height of each bob
    y1 = l2 + l1 * (1 - jnp.cos(phi1))  # set phi=0 position to zero potential energy
    y2 = y1 - l2 * jnp.cos(phi2)
    # y2 = l1 * (1 - jnp.cos(phi1)) + l2 *(1 - jnp.cos(phi2))
    V = m1 * g * y1 + m2 * g * y2
    return T+V


def HNN_dynamics(hamiltonian, can_state, t=0):
    # can_state = wrap_state(can_state)  # Force the first two angle coordinates in [-pi,pi). Delete for other examples.
    q, p = jnp.split(can_state, 2, axis=-1)
    q_t = jax.grad(hamiltonian, 1)(q, p)
    p_t = - jax.grad(hamiltonian, 0)(q, p)
    can_state_t = jnp.concatenate([q_t, p_t])
    return can_state_t


@partial(jax.jit, backend='cpu')
def general_analytical_odeint(x0, t):
    return odeint(analytical_dynamics, x0, t, rtol=1e-12, atol=1e-12)


def simple_odeint(dynamics_fun, state0, t, num_updates=1):
    x = state0
    X = []
    tp_last = 0.
    for tp in t:
        dt = tp - tp_last
        dx = rk4_update(dynamics_fun, x, num_updates, dt)
        x += dx
        X.append(x)
        tp_last = tp
    return jnp.array(X)  # jnp.array necessary to vmap the function


def rk4_update(dynamics_fun, state, num_updates, delta_t, t=None):
    """Applies num_update Runge-Kutta4 steps to integrate over delta_t. Returns update: delta_state"""
    def get_update(update):
        dt = delta_t / num_updates
        current_state = state + update
        k1 = dt * dynamics_fun(current_state)
        k2 = dt * dynamics_fun(current_state + k1 / 2)
        k3 = dt * dynamics_fun(current_state + k2 / 2)
        k4 = dt * dynamics_fun(current_state + k3)
        return update + 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    state_update = 0
    for _ in range(num_updates):
        state_update = get_update(state_update)
    return state_update  # jnp.array(state_update)


# Path:
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_path = THIS_DIR + "/eval/"
listEVAL = sorted(os.listdir(eval_path))
print("Cases: " + str(listEVAL))
# BaselineFile = "/doublePendulum_HNN_B_2x128_Data4x1500_params_for_loss_0.022198114544153214.pkl"  # TODO: BaselineFile
BaselineFile = "/doublePendulum_HNN_B_2x128_Data4x150_params_for_loss_16.397838592529297.pkl"

labels = [r'Baseline 2x128', r'HNN 2x128', r'HNN+H-Reg. 2x128']  # TODO: labels r'Baseline 2x128',
print("Labels: " + str(labels))


def evaluate_deltaH():
    print("------------------------------                           ------------------------------")
    print("------------------------------ Evaluate Delta H mean/max ------------------------------")
    print("------------------------------                           ------------------------------")
    t = jnp.linspace(0, 100, 101)
    rng = jax.random.PRNGKey(7)
    X0phi = jax.random.uniform(rng, (10, 2), minval=-0.8 * jnp.pi, maxval=0.8 * jnp.pi)
    X0phit = jnp.zeros_like(X0phi)
    X0 = jnp.hstack((X0phi, X0phit))
    print("X0 for evaluate_deltaH: \n{}".format(X0))

    # Baseline:
    len_state = 4
    hidden_dim = 128  # 128
    output_dim = 4

    mlp_init_fun, mlp_apply_fun = stax.serial(
        stax.Dense(hidden_dim),
        stax.Softplus,
        stax.Dense(hidden_dim),
        stax.Softplus,
        stax.Dense(output_dim), )

    @jax.jit
    def Baseline_dynamics(params, state, t=0):
        state = wrap_state(state)
        state_t = mlp_apply_fun(params, state)
        return state_t

    print("Processing: " + str(BaselineFile))
    with open(THIS_DIR + BaselineFile, 'rb') as fp:
        case_dict = pickle.load(fp)
    params = case_dict['params']
    dynamics = jax.jit(partial(Baseline_dynamics, params))

    def Base_simple_odeint(x0, t):
        return simple_odeint(dynamics, x0, t, num_updates=100)

    def Base_odeint(x0, t):
        return odeint(dynamics, x0, t, rtol=1e-12, atol=1e-12)

    X = jax.vmap(Base_simple_odeint, (0, None))(X0, t)  # dim = num_initial_states * len_trajectory * 4
    # X = jax.vmap(Base_odeint, (0, None))(X0, t)
    h_pred = jax.vmap(jax.vmap(analytical_hamiltonian))(X)
    h_test = jax.vmap(analytical_hamiltonian)(X0)
    h_test = jnp.reshape(h_test, (len(X0), 1))

    delta_h = jnp.abs(h_pred - h_test)
    delta_h_mean = jnp.mean(jnp.abs(h_pred - h_test))
    delta_h_max = jnp.max(jnp.abs(h_pred - h_test))
    delta_h_std = jnp.std(jnp.abs(h_pred - h_test))
    normalization = 9.81 * 6 / 100
    print("normalized_delta_h_mean: {}".format(delta_h_mean / normalization))
    print("normalized_delta_h_max: {}".format(delta_h_max / normalization))
    print("normalized_delta_h_std: {}".format(delta_h_std / normalization))

    # HNN:
    len_state = 4
    hidden_dim = 128  # 128
    output_dim = 1

    mlp_init_fun, mlp_apply_fun = stax.serial(
        stax.Dense(hidden_dim),
        stax.Softplus,
        stax.Dense(hidden_dim),
        stax.Softplus,
        stax.Dense(output_dim), )

    def learned_hamiltonian(params):  # mlp_apply_fun
        def hamiltonian(q, p):
            state = jnp.concatenate([q, p])
            state = wrap_state(state)
            # squeeze because jax.grad only defined for scalar input shape: () NOT (1,)
            return jnp.squeeze(mlp_apply_fun(params, state))
        return hamiltonian

    for file in listEVAL:
        print("Processing: " + str(file))
        with open(eval_path + file, 'rb') as fp:
            case_dict = pickle.load(fp)
        params = case_dict['params']
        dynamics = jax.jit(partial(HNN_dynamics, learned_hamiltonian(params)))

        def HNN_simple_odeint(x0, t):
            return simple_odeint(dynamics, x0, t, num_updates=100)

        def HNN_odeint(x0, t):
            return odeint(dynamics, x0, t, rtol=1e-12, atol=1e-12)

        X = jax.vmap(HNN_simple_odeint, (0, None))(X0, t)  # dim = num_initial_states * len_trajectory * 4
        # X = jax.vmap(HNN_odeint, (0, None))(X0, t)
        h_pred = jax.vmap(jax.vmap(analytical_hamiltonian))(X)
        h_test = jax.vmap(analytical_hamiltonian)(X0)
        h_test = jnp.reshape(h_test, (len(X0), 1))

        delta_h = jnp.abs(h_pred - h_test)
        delta_h_mean = jnp.mean(jnp.abs(h_pred - h_test))
        delta_h_max = jnp.max(jnp.abs(h_pred - h_test))
        delta_h_std = jnp.std(jnp.abs(h_pred - h_test))
        normalization = 9.81 * 6 / 100
        print("normalized_delta_h_mean: {}".format(delta_h_mean / normalization))
        print("normalized_delta_h_max: {}".format(delta_h_max / normalization))
        print("normalized_delta_h_std: {}".format(delta_h_std / normalization))


if __name__ == "__main__":
    evaluate_deltaH()









