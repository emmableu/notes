import torch
def multi_head_attention(X: torch.tensor, H: int):
    B, T, C = X.shape
    D = C // H
    Wq, Wk, Wv = torch.randn(D,D),torch.randn(D,D),torch.randn(D,D)
    Q, K, V = Wq @ X, Wk @ X, Wv @ X
    Q = Q.view(B, T, H, D).transpose(1,2)
    K = K.view(B, T, H, D).transpose(1,2)
    V = V.view(B, T, H, D).transpose(1,2) # B H, T, D
    A = Q @ K.transpose() # A is B, H, T, T
    A = A * D ** (-0.5)
    mask = torch.tril(torch.ones(T, T))
    A = A.masked_fill(mask == 0, -torch.inf)
    A_prime = A.softmax(-1)
    out = A_prime @ V
    out = out.transpose(1,2).reshape(B, T, C) # cannot use view because the data is not contiguous memory
    return out
input = torch.tensor([[[0.1, 0.4], [0.6, 0.5], [0.2, 0.2]]])
output = multi_head_attention(input, 2)
print(output)