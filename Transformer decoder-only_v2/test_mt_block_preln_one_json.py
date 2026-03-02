import json
from pathlib import Path
import torch

from mt_attention_preln import MTEmbedding
from mt_block_preln import MusicTransformerBlockPreLN

# ====== CONFIG ======
JSON_PATH = Path(r"C:\Users\hersa\PycharmProjects\TFG_INFO\tokenizer\tokens_json_bpe\maestro-v3.0.0\2006\MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.json")
TOKEN_FIELD = "ids"
VOCAB_SIZE = 30000

T = 128
D_MODEL = 256
N_HEADS = 8
MAX_SEQ_LEN = 512
DROPOUT = 0.0     # test determinista
DEVICE = "cpu"
SEED = 123
# ====================

def main():
    torch.manual_seed(SEED)
    obj = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    ids = obj[TOKEN_FIELD]
    idx = torch.tensor(ids[:T], dtype=torch.long).unsqueeze(0).to(DEVICE)

    emb = MTEmbedding(VOCAB_SIZE, D_MODEL, dropout=DROPOUT, debug=True).to(DEVICE)
    block = MusicTransformerBlockPreLN(D_MODEL, N_HEADS, MAX_SEQ_LEN, dropout=DROPOUT, debug=True).to(DEVICE)

    emb.eval(); block.eval()
    with torch.no_grad():
        x = emb(idx)
        y, attn = block(x, return_attn=True)

    print(f"[OUT] x={tuple(x.shape)} y={tuple(y.shape)} attn={tuple(attn.shape)}")

    # check causal: futuro 0
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=DEVICE), diagonal=1)
    max_future = attn[0,0][future].max().item()
    print(f"[CHECK] max attn weight (future) = {max_future:.12f}")
    assert max_future < 1e-8

    print("[DONE] MT block (Pre-LN) OK ✅")

if __name__ == "__main__":
    main()