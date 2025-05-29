def debug_tensor_grads(tensor, name="tensor"):
    """
    Debug function to check gradient information for a tensor.

    Args:
        tensor: PyTorch tensor to inspect
        name: Name to display for the tensor (for identification)
    """
    print(f"=== {name} ===")
    print(f"  Shape: {tensor.shape}")
    print(f"  Device: {tensor.device}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  requires_grad: {tensor.requires_grad}")
    print(f"  is_leaf: {tensor.is_leaf}")
    print(f"  grad_fn: {tensor.grad_fn}")

    if tensor.grad is not None:
        print(f"  grad shape: {tensor.grad.shape}")
        print(f"  grad norm: {tensor.grad.norm().item():.6f}")
    else:
        print(f"  grad: None")

    print()
