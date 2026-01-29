import torch

def f(x):
    return torch.stack([
        x[0]**2 + x[1] * x[2],  # f1
        x[0] * x[1] + x[2]**2    # f2
    ])

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Compute function value
y = f(x)
print(f"f(x) = {y}")
jacobian = torch.autograd.functional.jacobian(f, x)
print(f"jacobian = {jacobian}")
print(f"\njacobian shape: {jacobian.shape}")

v = torch.tensor([1.0, 0.5])
y = f(x)
y.backward(v)
vjp = x.grad

print(f"\nVJP (v^T J) = {vjp}")  # shape [3]
print(f"Manual: {v @ jacobian}")