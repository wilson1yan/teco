import jax
import jax.numpy as jnp

@jax.custom_vjp
def identity(x):
    return x

def identity_fwd(x):
    return x, None

def identity_bwd(_, g):
    return g,

identity.defvjp(identity_fwd, identity_bwd)


@jax.custom_vjp
def f_pmean(x):
    return x

def f_pmean_fwd(x):
    return f_pmean(x), None

def f_pmean_bwd(_, g):
    return jax.lax.pmean(g, 'model'), 


# identity in forward pass, psum in backward
@jax.custom_vjp
def f_psum(x):
    return x


def f_psum_fwd(x):
    return f_psum(x), None


def f_psum_bwd(_, g):
    return jax.lax.psum(g, "model"),


f_psum.defvjp(f_psum_fwd, f_psum_bwd)


# identity in forward pass, pmean in backward
@jax.custom_vjp
def f_pmean(x):
    return x


def f_pmean_fwd(x):
    return f_psum(x), None


def f_pmean_bwd(_, g):
    return jax.lax.pmean(g, "model"),


f_pmean.defvjp(f_pmean_fwd, f_pmean_bwd)


# psum in forward pass, identity in backward
@jax.custom_vjp
def g_psum(x):
    return jax.lax.psum(x, "model")


def g_psum_fwd(x):
    return g_psum(x), None


def g_psum_bwd(_, g):
    return g,


g_psum.defvjp(g_psum_fwd, g_psum_bwd)


# all_gather

def create_g_all_gather(axis=0):
    assert axis >= 0, f"Does not work with axis: {axis} < 0"

    @jax.custom_vjp
    def g_all_gather(x):
        return jax.lax.all_gather(x, 'model', axis=axis)

    def g_all_gather_fwd(x):
        return g_all_gather(x), None

    def g_all_gather_bwd(_, g):
        g = jax.lax.dynamic_slice_in_dim(g, jax.lax.axis_index('model'), 1, axis=axis)
        g = jnp.squeeze(g, axis=axis)
        return g,

    g_all_gather.defvjp(g_all_gather_fwd, g_all_gather_bwd)
    return g_all_gather


def create_g_index_select(axis=0):
    assert axis >= 0, f"Does not work with axis: {axis} < 0"

    @jax.custom_vjp
    def g_index_select(x):
        x = jax.lax.dynamic_slice_in_dim(x, jax.lax.axis_index('model'), 1, axis=axis)
        x = jnp.squeeze(x, axis=axis)
        return x

    def g_index_select_fwd(x):
        return g_index_select(x), None

    def g_index_select_bwd(_, g):
        g = jax.lax.all_gather(g, 'model', axis=axis)
        return g,
    
    g_index_select.defvjp(g_index_select_fwd, g_index_select_bwd)
    return g_index_select


