import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from functools import partial
import matplotlib.pyplot as plt
import pickle


# from jax.lib import xla_bridge
# print(xla_bridge.get_backend().platform)
#
# from google.colab import drive
# drive.mount('/content/gdrive')


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


# def cartesian2radial_x(x):
#     x1, y1, x2, y2, x1_t, y1_t, x2_t, y2_t = x
#     # assert(jnp.isclose(x1 ** 2 + y1 ** 2, l1) and jnp.isclose((x2-x1) ** 2 + (y2-y1) ** 2, l2))
#     phi1 = jnp.arctan2(x1, -y1)
#     phi2 = jnp.arctan2((x2-x1), -(y2-y1))
#     phi1_t = (y1_t * x1 - x1_t * y1)/(x1**2 + y1**2)
#     phi2_t = ((y2_t-y1_t) * (x2-x1) - (x2_t-x1_t) * (y2-y1))/((x2-x1)**2 + (y2-y1)**2)
#     return jnp.array([phi1, phi2, phi1_t, phi2_t])


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


# def cartesian2radial_x_t(x_t, x):
#     x1, y1, x2, y2, x1_t, y1_t, x2_t, y2_t = x
#     x1_t, y1_t, x2_t, y2_t, x1_tt, y1_tt, x2_tt, y2_tt = x_t
#     phi1_t = (y1_t * x1 - x1_t * y1)/(x1**2 + y1**2)
#     phi2_t = ((y2_t-y1_t) * (x2-x1) - (x2_t-x1_t) * (y2-y1))/((x2-x1)**2 + (y2-y1)**2)
#     phi1_tt = (x1*y1_tt - x1_tt*y1)/(x1**2 + y1**2)
#     phi2_tt = ((x2-x1)*(y2_tt-y1_tt) - (x2_tt-x1_tt)*(y2-y1))/((x2-x1)**2 + (y2-y1)**2)
#     return jnp.array([phi1_t, phi2_t, phi1_tt, phi2_tt])


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
    return T+V


def wrap_state(state):
    # wrap generalized coordinates to [-pi, pi)
    phi1 = (state[0] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    phi2 = (state[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    phi1_t = state[2]
    phi2_t = state[3]
    return jnp.array([phi1, phi2, phi1_t, phi2_t])


def plot_hamiltonian_IdMap(Xt, Xt_pred, figname):
    Fig, Axes = plt.subplots(1, 4, figsize=(14, 3.5), dpi=120)
    Axes[0].set_title('Predicting $\dot {p1_1}$')
    Axes[0].set_xlabel('$\dot {p1_1}$ actual')
    Axes[0].set_ylabel('$\dot {p1_1}$ predicted')
    Axes[0].scatter(Xt[:, 4], Xt_pred[:, 4], s=3, alpha=0.2)
    Axes[0].set_aspect('equal', 'box')

    Axes[1].set_title('Predicting $\dot {p1_2}$')
    Axes[1].set_xlabel('$\dot {p1_2}$ actual')
    Axes[1].set_ylabel('$\dot {p1_2}$ predicted')
    Axes[1].scatter(Xt[:, 5], Xt_pred[:, 5], s=3, alpha=0.2)
    Axes[1].set_aspect('equal', 'box')

    Axes[2].set_title('Predicting $\dot {p2_1}$')
    Axes[2].set_xlabel('$\dot {p2_1}$ actual')
    Axes[2].set_ylabel('$\dot {p2_1}$ predicted')
    Axes[2].scatter(Xt[:, 6], Xt_pred[:, 6], s=3, alpha=0.2)
    Axes[2].set_aspect('equal', 'box')

    Axes[3].set_title('Predicting $\dot {p2_2}$')
    Axes[3].set_xlabel('$\dot {p2_2}$ actual')
    Axes[3].set_ylabel('$\dot {p2_2}$ predicted')
    Axes[3].scatter(Xt[:, 7], Xt_pred[:, 7], s=3, alpha=0.2)
    Axes[3].set_aspect('equal', 'box')

    Fig.set_tight_layout(True)
    Fig.savefig(figname)
    Fig.show()


def plot_loss_curve(train_losses, val_losses, figname="loss_curve"):
    plt.figure(figsize=(8, 3.5), dpi=120)
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')
    plt.yscale('log')
    plt.ylim(None, 200)
    plt.title('Losses over training')
    plt.xlabel("Train step")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.savefig(figname)
    plt.show()


def analytical_H(z, t=0, m1=1, m2=1, l1=1, l2=1, g=9.81):
    # only valid for m1 = m2 = 1.
    q, p = jnp.split(z, 2, axis=-1)
    y1 = l1+l2+q[1]
    y2 = l1+l2+q[3]
    return 0.5 * jnp.dot(p, p) + m1 * g * y1 + m2 * g * y2


def CHNN_dynamics(H, z, t=0):
    # only valid for m1 = m2 = 1.
    q1, q2, p1, p2 = jnp.split(z, 4, axis=-1)

    Dphi1 = jnp.block([[2 * q1.reshape((2, 1)), 2 * p1.reshape((2, 1))],
                       [jnp.zeros((2, 2))],
                       [jnp.zeros((2, 1)), 2 * q1.reshape((2, 1))],
                       [jnp.zeros((2, 2))]])

    Dphi2 = jnp.block([[-2 * (q2-q1).reshape((2, 1)), -2 * (p2-p1).reshape((2, 1))],
                       [2 * (q2-q1).reshape((2, 1)), 2 * (p2-p1).reshape((2, 1))],
                       [jnp.zeros((2, 1)), -2 * (q2-q1).reshape((2, 1))],
                       [jnp.zeros((2, 1)), 2 * (q2-q1).reshape((2, 1))]])

    Dphi = jnp.hstack((Dphi1, Dphi2))
    J = jnp.block([[jnp.zeros((4, 4)), jnp.eye(4)],
                   [- jnp.eye(4), jnp.zeros((4, 4))]])

    P = jnp.eye(8) - J @ Dphi @ jnp.linalg.solve(Dphi.T @ J @ Dphi, Dphi.T)

    return P @ J @ jax.grad(H)(z)  # = z_t


def learned_H(params):  # mlp_apply_fun
    @jax.jit
    def H(z):
        return jnp.squeeze(mlp_apply_fun(params, z))
    return H


def get_general_odeint_dataset(X0, t, L1, L2):
    @partial(jax.jit, backend='cpu')
    def general_analytical_odeint(x0, t, l1, l2):
        x = odeint(partial(analytical_dynamics, l1=l1, l2=l2), x0, t, rtol=1e-10, atol=1e-10)
        xt = jax.vmap(partial(analytical_dynamics, l1=l1, l2=l2))(x)
        h = jax.vmap(partial(analytical_hamiltonian, l1=l1, l2=l2))(x)

        z = jax.vmap(partial(radial2cartesian_x, l1=l1, l2=l2))(x)
        zt = jax.vmap(partial(radial2cartesian_x_t, l1=l1, l2=l2), (0, 0))(xt, x)

        # h = jax.vmap(partial(analytical_H, l1=l1, l2=l2))(z)
        return x, xt, h, z, zt

    X, XT, H, Z, ZT = jax.vmap(general_analytical_odeint, (0, None, 0, 0))(X0, t, L1, L2)

    return X, XT, H, Z, ZT


# X0 = jnp.array([[0.2*jnp.pi, 0.2*jnp.pi, 0, 0],
#                 [0.4*jnp.pi, 0.4*jnp.pi, 0, 0],
#                 [0.6*jnp.pi, 0.6*jnp.pi, 0, 0],
#                 [0.8*jnp.pi, 0.8*jnp.pi, 0, 0]], dtype=jnp.float32)  # 4*1500=6000
#
# L1 = jnp.ones((4,))
# L2 = jnp.ones((4,))


X0 = jnp.array([[0.2*jnp.pi, 0.2*jnp.pi, 0, 0],
                [0.4*jnp.pi, 0.4*jnp.pi, 0, 0],
                [0.6*jnp.pi, 0.6*jnp.pi, 0, 0],
                [0.8*jnp.pi, 0.8*jnp.pi, 0, 0]], dtype=jnp.float32)  # 4*1500=6000
N = 4
X0 = jnp.repeat(X0, N, axis=0)
prn1, prn2 = jax.random.split(jax.random.PRNGKey(0), 2)
L1 = jax.random.uniform(prn1, (4*N,), minval=0.2, maxval=2.2)
L2 = jax.random.uniform(prn2, (4*N,), minval=0.2, maxval=2.2)  # 4*4*1500=24000

# N = 100
# X0 = jnp.repeat(X0, N, axis=0)
# prn1, prn2 = jax.random.split(jax.random.PRNGKey(0), 2)
# L1 = jax.random.uniform(prn1, (4*N,), minval=0.2, maxval=2.2)
# L2 = jax.random.uniform(prn2, (4*N,), minval=0.2, maxval=2.2)
# t = jnp.linspace(0, 15, 16, dtype=jnp.float32)
# t_test = jnp.linspace(15, 30, 16, dtype=jnp.float32)


t = jnp.linspace(0, 150, 1501, dtype=jnp.float32)
t_test = jnp.linspace(150, 300, 1501, dtype=jnp.float32)

# %time x_train, xt_train, h_train, z_train, zt_train = jax.device_get(get_general_odeint_dataset(X0, t, L1, L2))
# %time x_test, xt_test, h_test, z_test, zt_test = jax.device_get(get_general_odeint_dataset(X0, t_test, L1, L2))

x_train, xt_train, h_train, z_train, zt_train = jax.device_get(get_general_odeint_dataset(X0, t, L1, L2))
x_test, xt_test, h_test, z_test, zt_test = jax.device_get(get_general_odeint_dataset(X0, t_test, L1, L2))

x_train = jax.device_get(jnp.concatenate(x_train))
xt_train = jax.device_get(jnp.concatenate(xt_train))
x_test = jax.device_get(jnp.concatenate(x_test))
xt_test = jax.device_get(jnp.concatenate(xt_test))
h_train = jax.device_get(jnp.concatenate(h_train))
h_test = jax.device_get(jnp.concatenate(h_test))
z_train = jax.device_get(jnp.concatenate(z_train))
zt_train = jax.device_get(jnp.concatenate(zt_train))
z_test = jax.device_get(jnp.concatenate(z_test))
zt_test = jax.device_get(jnp.concatenate(zt_test))

z_train = jax.device_put(z_train)
z_test = jax.device_put(z_test)

num_epochs = 150001  # 150001
log_every_epochs = 1000
lr1 = 1e-2
lr2 = 1e-3
lr3 = 1e-4
hreg = 0.0

len_state = 8
hidden_dim = 128
output_dim = 1

mlp_init_fun, mlp_apply_fun = stax.serial(
    stax.Dense(hidden_dim),
    stax.Softplus,
    stax.Dense(hidden_dim),
    stax.Softplus,
    stax.Dense(output_dim), )

opt_init, opt_update, get_params = optimizers.adam(
    lambda t: jnp.select([t < (num_epochs // 3),
                          t < (2 * num_epochs // 3),
                          t >= (2 * num_epochs // 3)],
                          [lr1, lr2, lr3]))

prng_key = jax.random.PRNGKey(0)
output_shape, init_params = mlp_init_fun(prng_key, (-1, len_state))
opt_state = opt_init(init_params)


@jax.jit
def loss_fun(params, batch):
    states, targets = batch
    preds = jax.vmap(
        partial(CHNN_dynamics, learned_H(params))
                    )(states)
    return jnp.mean(jnp.square(preds - targets))  # jnp.abs


@jax.jit
def loss_fun_hreg(params, batch, hreg):
    states, targets, h_targets = batch
    preds = jax.vmap(
        partial(CHNN_dynamics, learned_H(params))
                    )(states)
    h_preds = jax.vmap(learned_H(params))(states)
    return jnp.mean(jnp.square(preds - targets)) + hreg * jnp.mean(jnp.square(h_preds - h_targets))  # jnp.abs


@jax.jit
def step(e, opt_state, batch):
    params = get_params(opt_state)
    grad = jax.grad(loss_fun)(params, batch)
    return opt_update(e, grad, opt_state)


@jax.jit
def step_hreg(e, opt_state, batch, hreg):
    params = get_params(opt_state)
    grad = jax.grad(loss_fun_hreg)(params, batch, hreg)
    return opt_update(e, grad, opt_state)


# %%time
train_losses = []
val_losses = []
val_loss = 0.0
for epoch in range(num_epochs):
    opt_state = step_hreg(epoch, opt_state, (z_train, zt_train, h_train), hreg)
    if epoch % log_every_epochs == 0:
        train_loss = loss_fun_hreg(get_params(opt_state), (z_train, zt_train, h_train), 0.0)  # hreg
        val_loss = loss_fun_hreg(get_params(opt_state), (z_test, zt_test, h_test), 0.0)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print('[Epoch {:d}] train_loss: {:.5f} val_loss: {:.5f} '.format(epoch, train_loss, val_loss))

plot_loss_curve(train_losses, val_losses)
params = get_params(opt_state)  # best_params, best_loss


# %%time
# train_losses = []
# val_losses = []
# val_loss = 0.0
# for epoch in range(num_epochs):
#     opt_state = step(epoch, opt_state, (x_train, xt_train))
#     if epoch % log_every_epochs == 0:
#         train_loss = loss_fun(get_params(opt_state), (x_train, xt_train))
#         val_loss = loss_fun(get_params(opt_state), (x_test, xt_test))
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         print('[Epoch {:d}] train_loss: {:.5f} val_loss: {:.5f} '.format(epoch, train_loss, val_loss))
#
# plot_loss_curve(train_losses, val_losses)
# params = get_params(opt_state)


path = F"/content/gdrive/MyDrive/ColabThesis/ColabThesisData/"
# path = F""

pickle_name = path + "doublePendulum_CHNN_Hreg0_2x128_Data4x1500_params_for_loss_{}.pkl".format(val_loss)
with open(pickle_name, "wb") as pa:
    pickle.dump({'params': params}, pa)

zt_pred = jax.vmap(
        partial(CHNN_dynamics, learned_H(params))
                  )(z_test)
plot_hamiltonian_IdMap(zt_test, zt_pred, figname=path + "doublePendulum_CHNN_Hreg0_2x128_Data4x1500_IdMap.png")

