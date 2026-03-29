import torch
from src.model.blocks import skew  # ajusta import si hace falta

def naive(q, Er):
    # q: (B,H,T,Dh), Er: (H,2T-1,Dh)
    B,H,T,Dh = q.shape
    out = torch.zeros(B,H,T,T, device=q.device, dtype=q.dtype)
    for i in range(T):
        for j in range(T):
            r = j - i + (T - 1)  # 0..2T-2
            out[:,:,i,j] = (q[:,:,i,:] * Er[:,r,:]).sum(dim=-1)
    return out

def main():
    torch.manual_seed(0)
    B,H,T,Dh = 2, 4, 8, 16
    q = torch.randn(B,H,T,Dh)
    Er = torch.randn(H,2*T-1,Dh)

    QEr = torch.einsum("bhtd,hrd->bhtr", q, Er)
    s1 = skew(QEr)
    s2 = naive(q, Er)
    diff = (s1 - s2).abs().max().item()
    print("max|diff| =", diff)
    assert diff < 1e-5

if __name__ == "__main__":
    main()