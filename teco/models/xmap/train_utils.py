from typing import Any
from flax.training import train_state


class TrainState(train_state.TrainState):
    model_state: Any


def create_xmap_train_state_spec(model, train_state):
    spec = model.model_spec.spec()

    return TrainState(
        step=(...,),
        apply_fn=model.apply,
        params=spec,
        tx=train_state.tx,
        opt_state=(
            type(train_state.opt_state[0])(count=(...,), mu=spec, nu=spec),
            (...,),
            (...,)
        ),
        model_state=(...,)
    )


def shard_train_state(model, train_state):
    model_spec = model.model_spec
    num_shards = model.config.num_shards

    adam_state = train_state.opt_state[0]

    return TrainState(
        step=train_state.step,
        apply_fn=model.apply,
        params=model_spec.shard(train_state.params, num_shards),
        tx=train_state.tx,
        opt_state=(
            type(adam_state)(
                count=adam_state.count,
                mu=model_spec.shard(adam_state.mu, num_shards),
                nu=model_spec.shard(adam_state.nu, num_shards)
            ),
            *train_state.opt_state[1:]
        ),
        model_state=train_state.model_state
    )


def unshard_train_state(model, train_state):
    model_spec = model.model_spec

    adam_state = train_state.opt_state[0]

    return TrainState(
        step=train_state.step,
        apply_fn=train_state.apply_fn,
        params=model_spec.unshard(train_state.params),
        tx=train_state.tx,
        opt_state=(
            type(adam_state)(
                count=adam_state.count,
                mu=model_spec.unshard(adam_state.mu),
                nu=model_spec.unshard(adam_state.nu)
            ),
            *train_state.opt_state[1:]
        ),
        model_state=train_state.model_state
    )
