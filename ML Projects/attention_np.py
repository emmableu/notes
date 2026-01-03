"""
post-ln
x => mha(x) => +(resnet) => layernorm(x)  =>  ffn(x) => + (resnet) => layernorm
   ---------->                           ------------>


x = mha(x) + x
x = layernorm(x)
x = ffn(x) + x
x = layernorm(x)


pre-ln
x => layernorm(x) => mha(x) => +    => layernorm =>  ffn(x) => +
   -------------------------->       -------------------------->

x = mha(layernorm(x)) + x
x = ffn(layernorm(x)) + x


B*T*C
[[1,3,2]]
banana chocolate cake
[[[0.1,-0.1], [0.1, 0.2], [0.14,0;2]]]


Q [[1,3,2]]
K [[1,3,2]]
1 3 2 * 1, 2 3

=> 1, 3, 3
=> 18 *, 9 +, variance = 2
"""
import torch
def self_attention(X: torch.tensor):
    B, T, C = X.shape
    Wq, Wk, Wv = torch.randn(C, C), torch.randn(C, C), torch.randn(C, C)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv # BTC

    A = Q @ K.tranpose(-1, -2) # dot product
    #     mask out A
    A = A * C ** (-0.5)
    A = A.softmax(dim=-1) # each query, will need all keys add up to 1 # attention matrix (B T T)
    # can add dropout

    out = A @ V

    Wo = torch.randn(C,C)
    out = out @ Wo
    # can add dropout

    return out

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention (no cross-attention).
    Input:  x of shape (B, T, d_model)
    Output: y of shape (B, T, d_model)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # One linear to get Q,K,V together is common and efficient
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        attn_mask (optional):
          - shape (B, T) where 1 means keep and 0 means mask out (padding mask), OR
          - shape (B, 1, 1, T) broadcastable to attention scores, OR
          - shape (B, T, T) / (B, 1, T, T) for explicit masks.
        causal:
          - if True, applies a causal mask preventing attending to future tokens.
        """
        B, T, C = x.shape
        assert C == self.d_model

        # (B, T, 3*d_model) -> split -> each (B, T, d_model)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to heads: (B, T, d_model) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # attention scores: (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply causal mask if needed (mask out j > i)
        if causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Apply provided attention mask if any
        if attn_mask is not None:
            # Common case: padding mask (B, T) with 1=keep, 0=mask
            if attn_mask.dim() == 2 and attn_mask.shape == (B, T):
                # broadcast to (B, 1, 1, T)
                key_mask = attn_mask[:, None, None, :].to(torch.bool)
                scores = scores.masked_fill(~key_mask, float("-inf"))
            else:
                # Make sure it's broadcastable to (B, n_heads, T, T)
                m = attn_mask.to(torch.bool)
                # Convention: True means keep. If your mask is 0/1, convert accordingly before passing.
                scores = scores.masked_fill(~m, float("-inf"))

        # attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # weighted sum: (B, n_heads, T, d_head)
        y = attn @ v

        # merge heads: -> (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # output projection
        y = self.out(y)
        y = self.resid_drop(y)
        return y
