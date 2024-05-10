import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch.fft import rfft, irfft
from dataclasses import dataclass
from efficient_kan import KAN

@dataclass
class Config:
    dim: int = 256
    hidden: int = 1024
    dss_dim: int = 256
    dss_hidden: int = 128
    vocab_size: int = 32000
    layers: int = 4
    dropout: float = 0.1

class DSS(nn.Module):
    def __init__(self, dss_hidden: int, dss_dim: int):
        super().__init__()

        self.norm = nn.LayerNorm(dss_hidden)
        self.LR = nn.Parameter(torch.randn(dss_dim))
        self.LI = nn.Parameter(torch.randn(dss_dim))
        self.CR = nn.Parameter(torch.randn(dss_hidden, dss_dim))
        self.CI = nn.Parameter(torch.randn(dss_hidden, dss_dim))
        self.D = nn.Parameter(torch.randn(dss_hidden))

    def forward(self, x):
        device = x.device
        seq_len = x.shape[1]

        x = self.norm(x)
        Lambda = -self.LR.exp() + 1j * self.LI.exp()
        C = self.CR + 1j * self.CI
        S = (Lambda.unsqueeze(-1) * torch.arange(seq_len, device=device).unsqueeze(0)).exp()
        C = C * (Lambda.exp() - 1) / Lambda
        SimDK = torch.einsum('hn, nl -> lh', C, S).real

        K_f = rfft(SimDK, n = seq_len * 2, dim = -2)
        u_f = rfft(x, n = seq_len * 2, dim = -2)
        y = irfft(u_f * K_f, seq_len * 2, dim = -2)[..., :seq_len, :]
        return y + self.D * x

class GSS(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
      
        self.config = config
        self.norm = nn.LayerNorm(config.dim)
        self.xg = KAN(config.dim, config.hidden, bias=False)
        self.dss_in = KAN(config.dim, config.dss_hidden, bias = False)
        self.dss = DSS(config.dss_hidden, config.dss_dim)
        self.gate = KAN(config.dss_hidden, config.hidden, bias = False)
        self.output = Linear(config.hidden, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, idx):
        shortcut = idx
        x = self.norm(idx)
        xg = self.xg(x)
        dss = self.dss_in(x)
        dss = self.gate(self.dss(dss))
        out = self.output(dss * xg)
        out = self.dropout(out)
        return out + shortcut

class KAN_GSS_LM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
      
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [GSS(config) for _ in range(config.layers)]
        )
        self.lm = Linear(config.dim, config.vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.lm(x)
        return x

    @torch.inference_mode()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.7, top_k = None):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1, :]
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            yield idx_next.item()
