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
    y1 = -l1 * jnp.cos(phi1)
    y2 = y1 - l2 * jnp.cos(phi2)
    V = m1 * g * y1 + m2 * g * y2
    return T+V


@jax.jit
def Baseline_dynamics(params, state, t=0):
    state = wrap_state(state)
    state_t = mlp_apply_fun(params, state)
    return state_t


def wrap_state(state):
    # wrap generalized coordinates to [-pi, pi)
    phi1 = (state[0] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    phi2 = (state[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    phi1_t = state[2]
    phi2_t = state[3]
    return jnp.array([phi1, phi2, phi1_t, phi2_t])


def plot_hamiltonian_IdMap(Xt, Xt_pred, figname):
    Fig, Axes = plt.subplots(1, 2)
    Axes[0].set_title('Predicting $\dot {p_1}$')
    Axes[0].set_xlabel('$\dot {p_1}$ actual')
    Axes[0].set_ylabel('$\dot {p_1}$ predicted')
    Axes[0].scatter(Xt[:, 2], Xt_pred[:, 2], s=3, alpha=0.2)
    Axes[0].set_aspect('equal', 'box')

    Axes[1].set_title('Predicting $\dot {p_2}$')
    Axes[1].set_xlabel('$\dot {p_2}$ actual')
    Axes[1].set_ylabel('$\dot {p_2}$ predicted')
    Axes[1].scatter(Xt[:, 3], Xt_pred[:, 3], s=3, alpha=0.2)
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


def get_general_odeint_dataset(X0, t):
    @partial(jax.jit, backend='cpu')
    def general_analytical_odeint(x0, t):
        return odeint(analytical_dynamics, x0, t, rtol=1e-11, atol=1e-11)
    X = jnp.concatenate(jax.vmap(general_analytical_odeint, (0, None))(X0, t))
    return X, jax.vmap(analytical_dynamics)(X), jax.vmap(analytical_hamiltonian)(X)


X0 = jnp.array([[0.2*jnp.pi, 0.2*jnp.pi, 0, 0],
                [0.4*jnp.pi, 0.4*jnp.pi, 0, 0],
                [0.6*jnp.pi, 0.6*jnp.pi, 0, 0],
                [0.8*jnp.pi, 0.8*jnp.pi, 0, 0]], dtype=jnp.float32)  # 4*1500=6000
t = jnp.linspace(0, 150, 1501, dtype=jnp.float32)
t_test = jnp.linspace(150, 300, 1501, dtype=jnp.float32)


# %time x_train, xt_train, h_train = jax.device_get(get_general_odeint_dataset(X0, t))
# %time x_test, xt_test, h_test = jax.device_get(get_general_odeint_dataset(X0, t_test))

x_train, xt_train, h_train = jax.device_get(get_general_odeint_dataset(X0, t))
x_test, xt_test, h_test = jax.device_get(get_general_odeint_dataset(X0, t_test))

x_train = jax.device_put(jax.vmap(wrap_state)(x_train))
x_test = jax.device_put(jax.vmap(wrap_state)(x_test))


num_epochs = 150001
log_every_epochs = 1000
lr1 = 1e-2
lr2 = 1e-3
lr3 = 1e-4


len_state = 4
hidden_dim = 128
output_dim = 4

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
        partial(Baseline_dynamics, params)
                    )(states)
    return jnp.mean(jnp.square(preds - targets))  # jnp.abs


@jax.jit
def step(e, opt_state, batch):
    params = get_params(opt_state)
    grad = jax.grad(loss_fun)(params, batch)
    return opt_update(e, grad, opt_state)


# %%time
train_losses = []
val_losses = []
val_loss = 0.0
for epoch in range(num_epochs):
    opt_state = step(epoch, opt_state, (x_train, xt_train))
    if epoch % log_every_epochs == 0:
        train_loss = loss_fun(get_params(opt_state), (x_train, xt_train))
        val_loss = loss_fun(get_params(opt_state), (x_test, xt_test))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print('[Epoch {:d}] train_loss: {:.5f} val_loss: {:.5f} '.format(epoch, train_loss, val_loss))

plot_loss_curve(train_losses, val_losses, figname="doublePendulum_HNN_B_loss_curve")
params = get_params(opt_state)


# path = F"/content/gdrive/MyDrive/ColabThesis/ColabThesisData/"
path = F""

pickle_name = path + "doublePendulum_HNN_B_2x128_Data4x1500_params_for_loss_{}.pkl".format(val_loss)
with open(pickle_name, "wb") as pa:
    pickle.dump({'params': params}, pa)

xt_pred = jax.vmap(
        partial(Baseline_dynamics, params)
                  )(x_test)
plot_hamiltonian_IdMap(xt_test, xt_pred, figname=path + "doublePendulum_HNN_B_2x128_Data4x1500_IdMap.png")







