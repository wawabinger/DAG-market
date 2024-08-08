import tenseal as ts
import numpy as np
from torch.linalg import norm

# Setup TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40


def enc_cos_similarity(x, y):
    norm_x = norm(x)
    norm_y = norm(y)

    party_x = x / norm_x
    party_y = y / norm_y

    plain1 = ts.plain_tensor(party_x)
    plain2 = ts.plain_tensor(party_y)

    encrypted_tensor1 = ts.ckks_tensor(context, plain1)
    ciphertext = encrypted_tensor1.dot(plain2)
    return ciphertext.decrypt().tolist()
