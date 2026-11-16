# Raises inside the kernel: expect status=crash with this traceback in `error`.
def custom_kernel(data):
    raise RuntimeError("deliberate failure from fixture")
