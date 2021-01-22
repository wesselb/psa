import jax
import jax.numpy as jnp
import lab as B
import wbml.out as out
from stheno.input import MultiInput
from stheno.jax import Measure, GP, EQ, Delta, cross
from varz.jax import Vars

from psa import psa_kl_estimator

B.epsilon = 1e-6
B.set_random_seed(1)
B.default_dtype = jnp.float32

# Setting of experiment:
x = B.linspace(0, 10, 500)
m = 2
p = 4
true_basis = Vars(jnp.float32).orthogonal(shape=(p, p))

# Build a model for the data.
prior = Measure()
z_model = [GP(0.95 * EQ() + 0.05 * Delta(), measure=prior) for _ in range(m)]
z_model += [GP(Delta(), measure=prior) for _ in range(p - m)]
y_model = [sum([true_basis[j, i] * z_model[i] for i in range(p)], 0) for j in range(p)]

# Sample some data.
y = B.concat(*prior.sample(*[p(x) for p in y_model]), axis=1)


def model(vs, z):
    """No model. Just check the entropy estimator."""
    return 0


# Construct PSA estimator.
vs = Vars(jnp.float32)
psa_objective = psa_kl_estimator(model, y, m)
psa_objective(vs)  # Initialise variables.


def target_objective(vs):
    """Model likelihood and the true entropy."""
    basis = vs["basis"]
    z_model = [sum([basis[i, j] * y_model[i] for i in range(p)], 0) for j in range(m)]
    entropy = cross(*z_model)(MultiInput(*[p(x) for p in z_model])).entropy()
    return -entropy - model(vs, y @ basis)


def grad_basis(f, vs):
    """Compute the gradient with respect to the basis for an estimator."""
    basis_latent = vs.get_vars("basis")[0]
    vs_copy = vs.copy()

    def to_diff(basis_latent_):
        vs_copy.assign("basis", basis_latent_, differentiable=True)
        return f(vs_copy)

    return jax.grad(to_diff)(basis_latent)


def norm(x):
    """L2 norm."""
    return B.sqrt(B.sum(x ** 2))


def cos_sim(x, y):
    """Cosine similarity."""
    return B.sum(x * y) / norm(x) / norm(y)


# Estimate gradient with PSA and true KL.
grad_psa = grad_basis(psa_objective, vs)
grad_true = grad_basis(target_objective, vs)

out.kv("PSA", grad_psa)
out.kv("True", grad_true)
out.kv("Diff.", grad_psa - grad_true)
out.kv("Diff. norm.", grad_psa / norm(grad_psa) - grad_true / norm(grad_true))
out.kv("Cosine sim.", cos_sim(grad_psa, grad_true))
