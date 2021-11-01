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


def plot_hamiltonian_IdMap(Xt, Xt_pred, figname):
    Fig, Axes = plt.subplots(1, 2)
    Axes[0].set_title('Predicting $\dot {q_1}$')
    Axes[0].set_xlabel('$\dot {q_1}$ actual')
    Axes[0].set_ylabel('$\dot {q_1}$ predicted')
    Axes[0].scatter(Xt[:, 0], Xt_pred[:, 0], s=3, alpha=0.2)
    Axes[0].set_aspect('equal', 'box')

    Axes[1].set_title('Predicting $\dot {p_1}$')
    Axes[1].set_xlabel('$\dot {p_1}$ actual')
    Axes[1].set_ylabel('$\dot {p_1}$ predicted')
    Axes[1].scatter(Xt[:, 1], Xt_pred[:, 1], s=3, alpha=0.2)
    Axes[1].set_aspect('equal', 'box')

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


def get_general_odeint_dataset(X0, t, L1):
    @partial(jax.jit, backend='cpu')
    def general_analytical_odeint(x0, t, l1):
        x = odeint(partial(analytical_dynamics, l1=l1), x0, t, rtol=1e-10, atol=1e-10)
        xt = jax.vmap(partial(analytical_dynamics, l1=l1))(x)
        h = jax.vmap(partial(analytical_hamiltonian, l1=l1))(x)

        z = jax.vmap(partial(radial2cartesian_x, l1=l1))(x)
        zt = jax.vmap(partial(radial2cartesian_x_t, l1=l1), (0, 0))(xt, x)

        # h = jax.vmap(partial(analytical_H, l1=l1, l2=l2))(z)
        return x, xt, h, z, zt

    X, XT, H, Z, ZT = jax.vmap(general_analytical_odeint, (0, None, 0))(X0, t, L1)

    return X, XT, H, Z, ZT


X0 = jnp.array([[0.2*jnp.pi, 0],
                [0.4*jnp.pi, 0],
                [0.6*jnp.pi, 0],
                [0.8*jnp.pi, 0]], dtype=jnp.float32)  # 4*1500=6000

# N = 4
# X0 = jnp.repeat(X0, N, axis=0)
# prn1, prn2 = jax.random.split(jax.random.PRNGKey(0), 2)
# L1 = jax.random.uniform(prn1, (4*N,), minval=0.2, maxval=2.2)

L1 = jnp.array([1., 1., 1., 1.])

# t = jnp.linspace(0, 150, 151, dtype=jnp.float32)
# t_test = jnp.linspace(150, 300, 151, dtype=jnp.float32)
t = jnp.linspace(0, 150, 16, dtype=jnp.float32)
t_test = jnp.linspace(150, 300, 16, dtype=jnp.float32)

# %time x_train, xt_train, h_train, z_train, zt_train = jax.device_get(get_general_odeint_dataset(X0, t, L1))
# %time x_test, xt_test, h_test, z_test, zt_test = jax.device_get(get_general_odeint_dataset(X0, t_test, L1))

x_train, xt_train, h_train, z_train, zt_train = jax.device_get(get_general_odeint_dataset(X0, t, L1))
x_test, xt_test, h_test, z_test, zt_test = jax.device_get(get_general_odeint_dataset(X0, t_test, L1))

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

# hreg = 0.0

len_state = 4
hidden_dim = 32
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


for hreg in [0.0, 0.01]:
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


    # path = F"/content/gdrive/MyDrive/ColabThesis/ColabThesisData/"
    path = F""

    pickle_name = path + "singlePendulum_CHNN_Hreg{}_2x32_Data4x16_params_for_loss_{}.pkl".format(hreg, val_loss)
    with open(pickle_name, "wb") as pa:
        pickle.dump({'params': params}, pa)

    zt_pred = jax.vmap(
            partial(CHNN_dynamics, learned_H(params))
                      )(z_test)
    plot_hamiltonian_IdMap(zt_test, zt_pred, figname=path + "singlePendulum_CHNN_Hreg{}_2x32_Data4x16_IdMap.png".format(hreg))




