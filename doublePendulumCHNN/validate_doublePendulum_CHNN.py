import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental.ode import odeint
from functools import partial
# import matplotlib.pyplot as plt
import pickle
import os


def radial2cartesian_x(x, l1=1., l2=1.):
    # Convert from radial to Cartesian coordinates.
    phi1, phi2, phi1_t, phi2_t = x

    x1 = l1 * jnp.sin(phi1)
    y1 = -l1 * jnp.cos(phi1)

    x2 = x1 + l2 * jnp.sin(phi2)
    y2 = y1 - l2 * jnp.cos(phi2)

    x1_t = l1 * jnp.cos(phi1) * phi1_t
    y1_t = l1 * jnp.sin(phi1) * phi1_t

    x2_t = x1_t + l2 * jnp.cos(phi2) * phi2_t
    y2_t = y1_t + l2 * jnp.sin(phi2) * phi2_t

    return jnp.array([x1, y1, x2, y2, x1_t, y1_t, x2_t, y2_t])


def cartesian2radial_x(x):
    x1, y1, x2, y2, x1_t, y1_t, x2_t, y2_t = x
    # assert(jnp.isclose(x1 ** 2 + y1 ** 2, l1) and jnp.isclose((x2-x1) ** 2 + (y2-y1) ** 2, l2))
    phi1 = jnp.arctan2(x1, -y1)
    phi2 = jnp.arctan2((x2 - x1), -(y2 - y1))
    phi1_t = (y1_t * x1 - x1_t * y1) / (x1 ** 2 + y1 ** 2)
    phi2_t = ((y2_t - y1_t) * (x2 - x1) - (x2_t - x1_t) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return jnp.array([phi1, phi2, phi1_t, phi2_t])


def radial2cartesian_x_t(x_t, x, l1=1., l2=1.):
    # Convert from radial to Cartesian coordinates.
    phi1, phi2, _, __ = x
    phi1_t, phi2_t, phi1_tt, phi2_tt = x_t

    x1_t = l1 * jnp.cos(phi1) * phi1_t
    y1_t = l1 * jnp.sin(phi1) * phi1_t

    x2_t = x1_t + l2 * jnp.cos(phi2) * phi2_t
    y2_t = y1_t + l2 * jnp.sin(phi2) * phi2_t

    x1_tt = l1 * (- jnp.sin(phi1) * phi1_t * phi1_t + phi1_tt * jnp.cos(phi1))
    y1_tt = l1 * (jnp.cos(phi1) * phi1_t * phi1_t + phi1_tt * jnp.sin(phi1))

    x2_tt = x1_tt + l2 * (-jnp.sin(phi2) * phi2_t * phi2_t + phi2_tt * jnp.cos(phi2))
    y2_tt = y1_tt + l2 * (jnp.cos(phi2) * phi2_t * phi2_t + phi2_tt * jnp.sin(phi2))
    return jnp.array([x1_t, y1_t, x2_t, y2_t, x1_tt, y1_tt, x2_tt, y2_tt])


def cartesian2radial_x_t(x_t, x):
    x1, y1, x2, y2, x1_t, y1_t, x2_t, y2_t = x
    x1_t, y1_t, x2_t, y2_t, x1_tt, y1_tt, x2_tt, y2_tt = x_t
    phi1_t = (y1_t * x1 - x1_t * y1) / (x1 ** 2 + y1 ** 2)
    phi2_t = ((y2_t - y1_t) * (x2 - x1) - (x2_t - x1_t) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    phi1_tt = (x1 * y1_tt - x1_tt * y1) / (x1 ** 2 + y1 ** 2)
    phi2_tt = ((x2 - x1) * (y2_tt - y1_tt) - (x2_tt - x1_tt) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return jnp.array([phi1_t, phi2_t, phi1_tt, phi2_tt])


def analytical_dynamics(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.81):
    """Analytical time derivative of the double pendulums state=(phi, phi2, phi1_t, phi2_t)"""
    phi1, phi2, phi1_t, phi2_t = state

    a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(phi1 - phi2)
    a2 = (l1 / l2) * jnp.cos(phi1 - phi2)

    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (phi2_t ** 2) * jnp.sin(phi1 - phi2) - \
         (g / l1) * jnp.sin(phi1)
    f2 = (l1 / l2) * (phi1_t ** 2) * jnp.sin(phi1 - phi2) - (g / l2) * jnp.sin(phi2)

    phi1_tt = (f1 - a1 * f2) / (1 - a1 * a2)
    phi2_tt = (f2 - a2 * f1) / (1 - a1 * a2)

    state_t = jnp.array([phi1_t, phi2_t, phi1_tt, phi2_tt])
    return state_t


def wrap_state(state):
    # wrap generalized coordinates to [-pi, pi)
    phi1 = (state[0] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    phi2 = (state[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    phi1_t = state[2]
    phi2_t = state[3]
    return jnp.array([phi1, phi2, phi1_t, phi2_t])


def canonical_analytical_hamiltonian(can_state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.81):  # only used for HNN Baseline
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
    V = m1 * g * y1 + m2 * g * y2
    return T + V


def analytical_hamiltonian(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.81):
    phi1, phi2, phi1_t, phi2_t = state

    y1 = l2 + l1 * (1 - jnp.cos(phi1))  # set phi=0 position to zero potential energy
    y2 = y1 - l2 * jnp.cos(phi2)
    V = m1 * g * y1 + m2 * g * y2
    # compute the kinetic energy of each bob
    K1 = 0.5 * m1 * (l1 * phi1_t) ** 2
    K2 = 0.5 * m2 * ((l1 * phi1_t) ** 2 + (l2 * phi2_t) ** 2 +
                     2 * l1 * l2 * phi1_t * phi2_t * jnp.cos(phi1 - phi2))
    T = K1 + K2
    return T + V


# def analytical_H(z, t=0, m1=1, m2=1, l1=1, l2=1, g=9.81):
#     # only valid for m1 = m2 = 1.
#     q, p = jnp.split(z, 2, axis=-1)
#     return 0.5 * jnp.dot(p, p) + m1 * g * q[1] + m2 * g * q[3]


def analytical_H(z, t=0, m1=1, m2=1, l1=1, l2=1, g=9.81):
    # only valid for m1 = m2 = 1.
    q, p = jnp.split(z, 2, axis=-1)
    y1 = l1 + l2 + q[1]
    y2 = l1 + l2 + q[3]
    return 0.5 * jnp.dot(p, p) + m1 * g * y1 + m2 * g * y2


def CHNN_dynamics(H, z, t=0):
    # only valid for m1 = m2 = 1.
    q1, q2, p1, p2 = jnp.split(z, 4, axis=-1)

    Dphi1 = jnp.block([[2 * q1.reshape((2, 1)), 2 * p1.reshape((2, 1))],
                       [jnp.zeros((2, 2))],
                       [jnp.zeros((2, 1)), 2 * q1.reshape((2, 1))],
                       [jnp.zeros((2, 2))]])

    Dphi2 = jnp.block([[-2 * (q2 - q1).reshape((2, 1)), -2 * (p2 - p1).reshape((2, 1))],
                       [2 * (q2 - q1).reshape((2, 1)), 2 * (p2 - p1).reshape((2, 1))],
                       [jnp.zeros((2, 1)), -2 * (q2 - q1).reshape((2, 1))],
                       [jnp.zeros((2, 1)), 2 * (q2 - q1).reshape((2, 1))]])

    Dphi = jnp.hstack((Dphi1, Dphi2))
    J = jnp.block([[jnp.zeros((4, 4)), jnp.eye(4)],
                   [- jnp.eye(4), jnp.zeros((4, 4))]])

    P = jnp.eye(8) - J @ Dphi @ jnp.linalg.solve(Dphi.T @ J @ Dphi, Dphi.T)

    return P @ J @ jax.grad(H)(z)  # = z_t


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
HNNBaselineFile = "/doublePendulum_HNN_Hreg0.2_2x128_Data4x1500_params_for_loss_0.0051595550030469894.pkl"  # TODO: BaselineFile
labels = [r'HNN+H-Reg 2x128', r'CHNN 2x128', r'CHNN+H-Reg 2x128']  # TODO: labels
print("Labels : " + str(labels))


def evaluate_deltaH():
    print("------------------------------                           ------------------------------")
    print("------------------------------ Evaluate Delta H mean/max ------------------------------")
    print("------------------------------                           ------------------------------")
    t = jnp.linspace(0, 100, 101)
    rng = jax.random.PRNGKey(7)  # 7
    X0phi = jax.random.uniform(rng, (10, 2), minval=-0.8 * jnp.pi, maxval=0.8 * jnp.pi)
    X0phit = jnp.zeros_like(X0phi)
    X0 = jnp.hstack((X0phi, X0phit))
    print("X0 for evaluate_deltaH: \n{}".format(X0))
    Z0 = jax.vmap(radial2cartesian_x)(X0)
    # CHNN:
    len_state = 8
    hidden_dim = 128  # 128
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
        normalization = 9.81 * 6 / 100
        print("normalized_delta_h_mean: {}".format(delta_h_mean / normalization))
        print("normalized_delta_h_max: {}".format(delta_h_max / normalization))
        print("normalized_delta_h_std: {}".format(delta_h_std / normalization))


if __name__ == "__main__":
    evaluate_deltaH()
