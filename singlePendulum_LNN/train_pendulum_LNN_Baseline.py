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


@jax.jit
def Baseline_dynamics(params, state, t=0):
    state = wrap_state(state)
    state_t = mlp_apply_fun(params, state)
    return state_t


def wrap_state(states):
    # wrap generalized coordinates to [-pi, pi]
    phi = (states[0] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    phi_t = states[1]
    return jnp.array([phi, phi_t])


def plot_lagrangian_IdMap(Xt, Xt_pred, figname="IdMap"):
    Fig, Axes = plt.subplots(1, 1)
    Axes.set_title('Predicting $\ddot {\phi_1}$')
    Axes.set_xlabel('$\ddot {\phi_1}$ actual')
    Axes.set_ylabel('$\ddot {\phi_1}$ predicted')
    Axes.scatter(Xt[:, 1], Xt_pred[:, 1], s=3, alpha=0.2)
    Axes.set_aspect('equal', 'box')
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
        return odeint(analytical_dynamics, x0, t, rtol=1e-12, atol=1e-12)
    X = jnp.concatenate(jax.vmap(general_analytical_odeint, (0, None))(X0, t))
    return X, jax.vmap(analytical_dynamics)(X)


X0 = jnp.array([[0.2*jnp.pi, 0],
                [0.4*jnp.pi, 0],
                [0.6*jnp.pi, 0],
                [0.8*jnp.pi, 0]], dtype=jnp.float32)  # 4*1500=6000

# t = jnp.linspace(0, 150, 151, dtype=jnp.float32)  # time steps 0 to N
# t_test = jnp.linspace(150, 300, 151, dtype=jnp.float32)  # time steps N to 2N

t = jnp.linspace(0, 150, 16, dtype=jnp.float32)  # time steps 0 to N
t_test = jnp.linspace(150, 300, 16, dtype=jnp.float32)  # time steps N to 2N

# %time x_train, xt_train = jax.device_get(get_general_odeint_dataset(X0, t)) # dynamics for first N time steps
# %time x_test, xt_test = jax.device_get(get_general_odeint_dataset(X0, t_test)) # dynamics for next N time steps

x_train, xt_train = jax.device_get(get_general_odeint_dataset(X0, t))
x_test, xt_test = jax.device_get(get_general_odeint_dataset(X0, t_test))

x_train = jax.device_put(jax.vmap(wrap_state)(x_train))
x_test = jax.device_put(jax.vmap(wrap_state)(x_test))


num_epochs = 150001
log_every_epochs = 1000
lr1 = 1e-2  # 1e-3
lr2 = 1e-3  # 3e-4
lr3 = 1e-4


len_state = 2
hidden_dim = 32
output_dim = 2

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

plot_loss_curve(train_losses, val_losses, figname="pendulum_LNN_B_loss_curve")
params = get_params(opt_state)


# path = F"/content/gdrive/MyDrive/ColabData/"
path = F""

pickle_name = path + "singlePendulum_LNN_B_2x32_Data4x16_params_for_loss_{}.pkl".format(val_loss)
with open(pickle_name, "wb") as pa:
    pickle.dump({'params': params}, pa)

xt_pred = jax.vmap(
        partial(Baseline_dynamics, params)
                  )(x_test)
plot_lagrangian_IdMap(xt_test, xt_pred, figname=path + "singlePendulum_LNN_B_2x32_Data4x16_IdMap.png")


