import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.ode import odeint
from functools import partial
# import matplotlib.pyplot as plt
import pickle
import os


def wrap_state(states):
    # wrap generalized coordinates to [-pi, pi]
    phi = (states[0] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    phi_t = states[1]
    return jnp.array([phi, phi_t])


def analytical_dynamics(state, t=0, m1=1, l1=1, g=9.81):
    """Analytical time derivative of the single pendulums state=(phi,phi_t)"""
    phi, phi_t = state
    phi_tt = -g/l1*jnp.sin(phi)
    state_t = jnp.array([phi_t, phi_tt])
    return state_t


def analytical_hamiltonian(state, t=0, m1=1, l1=1, g=9.81):
    phi, phi_t = state
    V = m1*g*l1*(1-jnp.cos(phi))
    T = 0.5*m1*l1**2*phi_t**2
    return T+V


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
BaselineFile = "/singlePendulum_LNN_B_2x32_Data4x150_params_for_loss_9.483709391133743e-07.pkl"  # TODO: BaselineFile
# BaselineFile = "/singlePendulum_LNN_B_2x32_Data4x16_params_for_loss_6.662805844825925e-07.pkl"

labels = [r'Baseline 2x32', r'LNN 2x32']  # TODO: labels
print("Labels: " + str(labels))


def evaluate_deltaH():
    print("------------------------------                           ------------------------------")
    print("------------------------------ Evaluate Delta H mean/max ------------------------------")
    print("------------------------------                           ------------------------------")

    t = jnp.linspace(0, 100, 101)
    rng = jax.random.PRNGKey(7)
    X0phi = jax.random.uniform(rng, (10, 1), minval=-0.8*jnp.pi, maxval=0.8*jnp.pi)
    X0phit = jnp.zeros_like(X0phi)
    X0 = jnp.hstack((X0phi, X0phit))
    print("X0: \n{}".format(X0))
    # Baseline:
    len_state = 2
    hidden_dim = 32  # 128
    output_dim = 2

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
    delta_h_std = jnp.std(jnp.abs(h_pred - h_test))
    delta_h_max = jnp.max(jnp.abs(h_pred - h_test))
    normalization = 9.81 * 2 / 100
    print("normalized_delta_h_mean: {}".format(delta_h_mean / normalization))
    print("normalized_delta_h_max: {}".format(delta_h_max / normalization))
    print("normalized_delta_h_std: {}".format(delta_h_std / normalization))

    # LNN:
    len_state = 2
    hidden_dim = 32  # 128
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
        # X = jax.vmap(LNN_odeint, (0, None))(X0, t)
        h_pred = jax.vmap(jax.vmap(analytical_hamiltonian))(X)
        h_test = jax.vmap(analytical_hamiltonian)(X0)
        h_test = jnp.reshape(h_test, (len(X0), 1))

        delta_h = jnp.abs(h_pred - h_test)
        delta_h_mean = jnp.mean(jnp.abs(h_pred - h_test))
        delta_h_max = jnp.max(jnp.abs(h_pred - h_test))
        delta_h_std = jnp.std(jnp.abs(h_pred - h_test))
        normalization = 9.81*2/100
        print("normalized_delta_h_mean: {}".format(delta_h_mean/normalization))
        print("normalized_delta_h_max: {}".format(delta_h_max/normalization))
        print("normalized_delta_h_std: {}".format(delta_h_std/normalization))


if __name__ == "__main__":
    evaluate_deltaH()
