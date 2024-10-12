from contextlib import contextmanager


@contextmanager
def disable_jaxtyping():
    """Temporarily disable Jaxtyping runtime checking."""
    import jaxtyping
    original_value = jaxtyping.config.jaxtyping_disable
    jaxtyping.config.jaxtyping_disable = True  # Disable Jaxtyping

    try:
        yield  # Execution inside the `with` block
    finally:
        jaxtyping.config.jaxtyping_disable = original_value  # Restore original value
