import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.ode import odeint
from functools import partial
# import matplotlib.pyplot as plt
import pickle
import os


def radial2cartesian_x(x, l1=1.):
    # Convert from radial to Cartesian coordinates.
    phi1, phi1_t = x

    x1 = l1 * jnp.sin(phi1)
    y1 = -l1 * jnp.cos(phi1)

    x1_t = l1 * jnp.cos(phi1) * phi1_t
    y1_t = l1 * jnp.sin(phi1) * phi1_t

    return jnp.array([x1, y1, x1_t, y1_t])


def radial2cartesian_x_t(x_t, x, l1=1.):
    # Convert from radial to Cartesian coordinates.
    phi1, _ = x
    phi1_t, phi1_tt = x_t

    x1_t = l1 * jnp.cos(phi1) * phi1_t
    y1_t = l1 * jnp.sin(phi1) * phi1_t

    x1_tt = l1 * (- jnp.sin(phi1) * phi1_t * phi1_t + phi1_tt * jnp.cos(phi1))
    y1_tt = l1 * (jnp.cos(phi1) * phi1_t * phi1_t + phi1_tt * jnp.sin(phi1))
    return jnp.array([x1_t, y1_t, x1_tt, y1_tt])


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


def analytical_H(z, t=0, m1=1, l1=1, g=9.81):
    q, p = jnp.split(z, 2, axis=-1)
    return 0.5/m1 * jnp.dot(p, p) + m1 * g * (q[1] + l1)


def CHNN_dynamics(H, z, t=0):
    m1 = 1.
    q, p = jnp.split(z, 2, axis=-1)
    Dphi = jnp.block([[2 * q.reshape((2, 1)), 2/m1 * p.reshape((2, 1))], [jnp.zeros((2, 1)), 2 * q.reshape((2, 1))]])
    J = jnp.block([[jnp.zeros((2, 2)), jnp.eye(2)],
                   [- jnp.eye(2), jnp.zeros((2, 2))]])

    P = jnp.eye(4) - J @ Dphi @ jnp.linalg.solve(Dphi.T @ J @ Dphi, Dphi.T)

    return P @ J @ jax.grad(H)(z)  # = z_t


def learned_H(params):  # mlp_apply_fun
    @jax.jit
    def H(z):
        return jnp.squeeze(mlp_apply_fun(params, z))
    return H


def cartesian2radial_x(x):
    x1, y1, x1_t, y1_t = x
    # assert(jnp.isclose(x1 ** 2 + y1 ** 2, l1) and jnp.isclose((x2-x1) ** 2 + (y2-y1) ** 2, l2))
    phi1 = jnp.arctan2(x1, -y1)
    phi1_t = (y1_t * x1 - x1_t * y1) / (x1 ** 2 + y1 ** 2)
    return jnp.array([phi1, phi1_t])


def cartesian2radial_x_t(x_t, x):
    x1, y1, x1_t, y1_t = x
    x1_t, y1_t, x1_tt, y1_tt = x_t
    phi1_t = (y1_t * x1 - x1_t * y1) / (x1 ** 2 + y1 ** 2)
    phi1_tt = (x1 * y1_tt - x1_tt * y1) / (x1 ** 2 + y1 ** 2)
    return jnp.array([phi1_t, phi1_tt])


def canonical_analytical_hamiltonian(hstate, t=0, m1=1, l1=1, g=9.81):
    phi, p = hstate
    T = p ** 2 / (2 * m1 * l1 ** 2)
    V = m1 * g * l1 * (1 - jnp.cos(phi))  # set phi=0 position to zero potential energy
    return T + V


def analytical_hamiltonian(state, t=0, m1=1, l1=1, g=9.81):
    phi, phi_t = state
    V = m1*g*l1*(1-jnp.cos(phi))
    T = 0.5*m1*l1**2*phi_t**2
    return T+V


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


@partial(jax.jit, backend='cpu')
def general_analytical_odeint(x0, t):
    return odeint(analytical_dynamics, x0, t, rtol=1e-12, atol=1e-12)


# Path:
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_path = THIS_DIR + "/eval/"
listEVAL = sorted(os.listdir(eval_path))
print("Cases: " + str(listEVAL))
HNNBaselineFile = "/singlePendulum_HNN_Hreg0.07_2x32_Data4x16_params_for_loss_2.5352193233629805e-07.pkl"  # TODO: BaselineFile
labels = [r'HNN+H-Reg 2x32', r'CHNN4x150 2x32', r'CHNN4x16 2x32', r'CHNN4x4x38 2x32', r'CHNN4x4x4 2x32']  # TODO: labels  , r'CHNN+H-Reg 2x32'
print("Labels : " + str(labels))


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
    Z0 = jax.vmap(radial2cartesian_x)(X0)

    # CHNN:
    len_state = 4
    hidden_dim = 32  # 128
    output_dim = 1
    mlp_init_fun, mlp_apply_fun = stax.serial(
        stax.Dense(hidden_dim),
        stax.Softplus,
        stax.Dense(hidden_dim),
        stax.Softplus,
        stax.Dense(output_dim), )

    def learned_H(params):  # mlp_apply_fun
        @jax.jit
        def H(z):
            return jnp.squeeze(mlp_apply_fun(params, z))
        return H

    for file in listEVAL:
        print("Processing: " + str(file))
        with open(eval_path + file, 'rb') as fp:
            case_dict = pickle.load(fp)
        params = case_dict['params']
        dynamics = jax.jit(partial(CHNN_dynamics, learned_H(params)))

        def CHNN_simple_odeint(z0, t):
            return simple_odeint(dynamics, z0, t, num_updates=100)

        def CHNN_odeint(z0, t):
            return odeint(dynamics, z0, t, rtol=1e-12, atol=1e-12)  # , rtol=1e-12, atol=1e-12

        Z = jax.vmap(CHNN_simple_odeint, (0, None))(Z0, t)  # dim = num_initial_states * len_trajectory * 4
        # Z = jax.vmap(CHNN_odeint, (0, None))(Z0, t)
        X = jax.vmap(jax.vmap(cartesian2radial_x))(Z)
        h_pred_Z = jax.vmap(jax.vmap(analytical_H))(Z)
        h_pred = jax.vmap(jax.vmap(analytical_hamiltonian))(X)
        print("analytical_H(z) =~ analytical_hamiltonian(x) :  {}".format(
            jnp.allclose(h_pred_Z, h_pred, atol=1e-3, rtol=1)))
        h_test = jax.vmap(analytical_hamiltonian)(X0)
        h_test = jnp.reshape(h_test, (len(X0), 1))

        delta_h = jnp.abs(h_pred - h_test)
        delta_h_mean = jnp.mean(jnp.abs(h_pred - h_test))
        delta_h_max = jnp.max(jnp.abs(h_pred - h_test))
        delta_h_std = jnp.std(jnp.abs(h_pred - h_test))
        normalization = 9.81 * 2 / 100
        print("normalized_delta_h_mean: {}".format(delta_h_mean / normalization))
        print("normalized_delta_h_max: {}".format(delta_h_max / normalization))
        print("normalized_delta_h_std: {}".format(delta_h_std / normalization))


if __name__ == "__main__":
    evaluate_deltaH()

