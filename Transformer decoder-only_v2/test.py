import torch
from mt_attention_preln import skew


def naive_srel(q, Er):
    """
    q:  (B,H,T,Dh)
    Er: (2T-1,Dh) con índice r=(j-i)+(T-1)
    """
    B, H, T, Dh = q.shape
    S = torch.zeros((B, H, T, T), dtype=q.dtype)
    for i in range(T):
        for j in range(T):
            r = (j - i) + (T - 1)
            S[:, :, i, j] = (q[:, :, i, :] * Er[r]).sum(dim=-1)
    return S


def main():
    torch.manual_seed(0)
    B, H, T, Dh = 1, 2, 8, 4

    q = torch.randn(B, H, T, Dh)
    Er = torch.randn(2 * T - 1, Dh)

    QEr = torch.matmul(q, Er.t())   # (B,H,T,2T-1)
    S_eff = skew(QEr)               # (B,H,T,T)
    S_ref = naive_srel(q, Er)

    diff = (S_eff - S_ref).abs().max().item()
    print(f"[OK] max|diff| skew vs naive = {diff:.10f}")
    assert diff < 1e-6, "Skew incorrecto (no coincide con naive)."


if __name__ == "__main__":
    main()