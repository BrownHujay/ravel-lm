import sys, time
sys.path.insert(0, r"C:\Users\newma\ravel-lm")
import torch
from ravel_lm.config import RavelConfig
from ravel_lm.model import RavelLM
from ravel_lm.runtime import FlatAdamW
import ravel_lm.triton_block as tb
from ravel_lm.deferred import flush_deferred

DEV = torch.device("cuda")
cfg = RavelConfig.from_json(r"C:\Users\newma\ravel-lm\configs\byte\ravel_3m_byte.json")
cfg.batch_size = 1; cfg.block_size = 2048; cfg.validate()

torch.manual_seed(0); m_ref = RavelLM(cfg).to(DEV)
torch.manual_seed(0); m_new = RavelLM(cfg).to(DEV)
torch.manual_seed(7)
x = torch.randint(0, cfg.vocab_size, (1, 2048), device=DEV)
y = torch.randint(0, cfg.vocab_size, (1, 2048), device=DEV)

# reference: force all triton_block paths off
orig_ok = tb.triton_ok
tb.triton_ok = lambda: False
out_ref = m_ref(x, y); out_ref["loss"].backward()
tb.triton_ok = orig_ok

out_new = m_new(x, y); out_new["loss"].backward()
flush_deferred()
ld = abs(float(out_ref["loss"]) - float(out_new["loss"]))
lg = (out_ref["logits"] - out_new["logits"]).abs().max().item()
worst = 0.0
for (nm, a), (_, b) in zip(m_ref.named_parameters(), m_new.named_parameters()):
    if a.grad is None or b.grad is None:
        if (a.grad is None) != (b.grad is None):
            print("GRAD MISSING:", nm); worst = float("inf")
        continue
    worst = max(worst, (a.grad - b.grad).abs().max().item())
print(f"loss diff {ld:.2e}  logits max {lg:.2e}  worst grad {worst:.2e}")

params = [p for p in m_new.parameters() if p.requires_grad]
opt = FlatAdamW([{"params": params, "weight_decay": 0.1}], lr=3e-4, betas=(0.9, 0.999))
def full():
    opt.zero_grad()
    m_new(x, y)["loss"].backward()
    opt.clip_grad_norm_(1.0)
    opt.step()
def t(fn, iters=50, warmup=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/iters*1000
ms = t(full)
print(f"full step: {ms:7.3f} ms  ({2048/ms*1000:,.0f} tokens/sec)")
