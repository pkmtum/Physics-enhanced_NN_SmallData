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


def analytical_dynamics(gen_state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.81):
    """Analytical time derivative of the double pendulums state=(phi, phi2, phi1_t, phi2_t)"""
    phi1, phi2, phi1_t, phi2_t = gen_state

    a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(phi1 - phi2)
    a2 = (l1 / l2) * jnp.cos(phi1 - phi2)

    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (phi2_t ** 2) * jnp.sin(phi1 - phi2) - \
         (g / l1) * jnp.sin(phi1)
    f2 = (l1 / l2) * (phi1_t ** 2) * jnp.sin(phi1 - phi2) - (g / l2) * jnp.sin(phi2)

    phi1_tt = (f1 - a1 * f2) / (1 - a1 * a2)
    phi2_tt = (f2 - a2 * f1) / (1 - a1 * a2)

    gen_state_t = jnp.array([phi1_t, phi2_t, phi1_tt, phi2_tt])
    return gen_state_t


def analytical_hamiltonian(gen_state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.81):
    phi1, phi2, phi1_t, phi2_t = gen_state
    # compute the height of each bob
    # y1 = -l1 * jnp.cos(phi1)
    y1 = l2 + l1 * (1 - jnp.cos(phi1))  # set phi=0 position to zero potential energy
    y2 = y1 - l2 * jnp.cos(phi2)
    V = m1 * g * y1 + m2 * g * y2
    # compute the kinetic energy of each bob
    K1 = 0.5 * m1 * (l1 * phi1_t) ** 2
    K2 = 0.5 * m2 * ((l1 * phi1_t) ** 2 + (l2 * phi2_t) ** 2 +
                     2 * l1 * l2 * phi1_t * phi2_t * jnp.cos(phi1 - phi2))
    T = K1 + K2
    return T + V


def LNN_dynamics(lagrangian, gen_state, t=0):
    # gen_state = wrap_state(gen_state)
    q, q_t = jnp.split(gen_state, 2, axis=-1)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
            @ (jax.grad(lagrangian, 0)(q, q_t)
               - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    gen_state_t = jnp.concatenate([q_t, q_tt])
    return gen_state_t


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
BaselineFile = "/doublePendulum_LNN_B_2x128_Data4x1500_params_for_loss_0.010443579405546188.pkl"  # TODO: BaselineFile
# BaselineFile = "/doublePendulum_LNN_B_2x128_Data4x150_params_for_loss_5.312648773193359.pkl"

labels = ['Baseline 2x128', 'LNN 2x128']  # TODO: labels
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

    # LNN:
    len_state = 4
    hidden_dim = 128  # 128
    output_dim = 1

    mlp_init_fun, mlp_apply_fun = stax.serial(
        stax.Dense(hidden_dim),
        stax.Softplus,
        stax.Dense(hidden_dim),
        stax.Softplus,
        stax.Dense(output_dim), )

    def learned_lagrangian(params):  # mlp_apply_fun
        @jax.jit
        def lagrangian(q, q_t):
            state = jnp.concatenate([q, q_t])
            state = wrap_state(state)
            return jnp.squeeze(mlp_apply_fun(params, state))
        return lagrangian

    for file in listEVAL:
        print("Processing: " + str(file))
        with open(eval_path + file, 'rb') as fp:
            case_dict = pickle.load(fp)
        params = case_dict['params']
        dynamics = jax.jit(partial(LNN_dynamics, learned_lagrangian(params)))

        def LNN_simple_odeint(x0, t):
            return simple_odeint(dynamics, x0, t, num_updates=100)

        def LNN_odeint(x0, t):
            return odeint(dynamics, x0, t, rtol=1e-12, atol=1e-12)

        X = jax.vmap(LNN_simple_odeint, (0, None))(X0, t)  # dim = num_initial_states * len_trajectory * 4
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

