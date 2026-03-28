import numpy as np

np.random.seed(42)

# Input: 2 tokens, each with 4 features (like a tiny batch)
x = np.array([
    [ 1.0, -0.5,  0.3, -1.2],
    [-0.8,  2.0, -0.1,  0.6],
])

# Weight matrices: expand dim=4 → hidden=6
W_fc  = np.random.randn(4, 6) * 0.5   # used by relu² (and W1 in SwiGLU)
W_fc2 = np.random.randn(4, 6) * 0.5   # only used by SwiGLU (the gate branch)
W_proj = np.random.randn(6, 4) * 0.5  # contracts hidden → dim

# --------------------------------------------------
# relu²
# --------------------------------------------------
h_relu = np.maximum(0, x @ W_fc)   # relu
h_relu2 = h_relu ** 2               # square it
out_relu2 = h_relu2 @ W_proj

print("=== relu² ===")
print("hidden activations (after relu²):\n", h_relu2.round(3))
print("output:\n", out_relu2.round(3))

# --------------------------------------------------
# SwiGLU
# --------------------------------------------------
def swish(z):
    return z * (1 / (1 + np.exp(-z)))   # z * sigmoid(z)

gate  = swish(x @ W_fc)   # gate branch: smooth nonlinearity
value = x @ W_fc2          # value branch: plain linear
h_swiglu = gate * value    # elementwise product — learned gating
out_swiglu = h_swiglu @ W_proj

print("\n=== SwiGLU ===")
print("gate  (swish branch):\n", gate.round(3))
print("value (linear branch):\n", value.round(3))
print("hidden activations (gate * value):\n", h_swiglu.round(3))
print("output:\n", out_swiglu.round(3))

# --------------------------------------------------
# Key difference: sparsity pattern
# --------------------------------------------------
print("\n=== How many hidden neurons are exactly zero? ===")
print(f"relu²:  {(h_relu2 == 0).sum()} of {h_relu2.size}")
print(f"SwiGLU: {(h_swiglu == 0).sum()} of {h_swiglu.size}  (swish never hits exactly 0)")
