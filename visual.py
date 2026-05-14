import numpy as np

_WINDOW_CASH = {}
_VIS_CACHE = {}


class _NoOpVis:
    """Silent stand-in when no Visdom server is available."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _vis(env='main'):
    if env not in _VIS_CACHE:
        try:
            from visdom import Visdom
            v = Visdom(env=env, raise_exceptions=True)
            # Verify the connection is actually alive
            if not v.check_connection(timeout_seconds=1):
                raise ConnectionError
            _VIS_CACHE[env] = v
        except Exception:
            _VIS_CACHE[env] = _NoOpVis()
    return _VIS_CACHE[env]


def visualize_image(tensor, name, label=None, env='main', w=250, h=250,
                    update_window_without_label=False):
    tensor = tensor.cpu()
    title = name + ('-{}'.format(label) if label is not None else '')

    _WINDOW_CASH[title] = _vis(env).image(
        tensor.numpy(), win=_WINDOW_CASH.get(title),
        opts=dict(title=title, width=w, height=h)
    )

    if update_window_without_label:
        _WINDOW_CASH[name] = _vis(env).image(
            tensor.numpy(), win=_WINDOW_CASH.get(name),
            opts=dict(title=name, width=w, height=h)
        )


def visualize_images(tensor, name, label=None, env='main', w=400, h=400,
                     update_window_without_label=False):
    tensor = tensor.cpu()
    title = name + ('-{}'.format(label) if label is not None else '')

    _WINDOW_CASH[title] = _vis(env).images(
        tensor.numpy(), win=_WINDOW_CASH.get(title), nrow=6,
        opts=dict(title=title, width=w, height=h)
    )

    if update_window_without_label:
        _WINDOW_CASH[name] = _vis(env).images(
            tensor.numpy(), win=_WINDOW_CASH.get(name), nrow=6,
            opts=dict(title=name, width=w, height=h)
        )


def visualize_scalar(scalar, name, iteration, env='main'):
    visualize_scalars(
        [scalar] if isinstance(scalar, float) or len(scalar) == 1 else scalar,
        [name], name, iteration, env=env
    )


def visualize_scalars(scalars, names, title, iteration, env='main'):
    assert len(scalars) == len(names)
    scalars, names = list(scalars), list(names)
    scalars = [s.cpu() if hasattr(s, 'cpu') else s for s in scalars]
    scalars = [s.numpy() if hasattr(s, 'numpy') else np.array([s]) for s in scalars]
    multi = len(scalars) > 1
    num = len(scalars)

    options = dict(
        fillarea=True,
        legend=names,
        width=400,
        height=400,
        xlabel='Iterations',
        ylabel=title,
        title=title,
        marginleft=30,
        marginright=30,
        marginbottom=80,
        margintop=30,
    )

    X = (
        np.column_stack(np.array([iteration] * num)) if multi else
        np.array([iteration] * num)
    )
    Y = np.column_stack(scalars) if multi else scalars[0]

    if title in _WINDOW_CASH:
        # updateTrace was removed in visdom >=0.1.9; use line(..., update='append')
        _vis(env).line(X=X, Y=Y, win=_WINDOW_CASH[title], opts=options, update='append')
    else:
        _WINDOW_CASH[title] = _vis(env).line(X=X, Y=Y, opts=options)
