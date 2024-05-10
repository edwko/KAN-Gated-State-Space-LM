
# Simple example of training and interface

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from kan_gss_lm import Config, KAN_GSS_LM
from tqdm import tqdm

@dataclass
class TrainConfig:
    lr: float = 0.00005
    block_size: int = 256
    batch_size: int = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_every_n_steps: int = 100
    update_desc_every_n_steps: int = 25
    max_iter: int = -1

    def use_cpu(self):
        self.device = torch.device('cpu')

class Data(Dataset):
    def __init__(self, data, cfg: TrainConfig):
        self.data = data
        self.block_size = cfg.block_size
        self.data_size = len(data) - (self.block_size * 2)
        self.device = cfg.device
        if self.data_size < self.block_size * 2:
            raise ValueError("Data is too short")

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        src = self.data[idx : idx + self.block_size]
        tgt = self.data[idx + 1 : idx + self.block_size + 1]
        src = torch.tensor(src, dtype=torch.int64).to(self.device)
        tgt = torch.tensor(tgt, dtype=torch.int64).to(self.device)
        return src, tgt

class Train():
    def __init__(self, model, dataset: Data, cfg: TrainConfig) -> None:
        self.dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.model_params()

    def _params_format(self, num):
        num = float('{:.3g}'.format(num))
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T', '-'][magnitude])

    def model_params(self):
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        pytorch_total_params_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\nTotal params:", self._params_format(pytorch_total_params), f"({pytorch_total_params})")
        print("Total trainable params:", self._params_format(pytorch_total_params_trainable), f"({pytorch_total_params_trainable})", "\n")

    def _step(self, src, tgt):
        self.optimizer.zero_grad()
        logits = self.model(src)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-1)
        loss.backward()
        self.optimizer.step()
        return loss

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pt")

    def train(self):
        self.model.train()
        first_loss = None
        eph = tqdm(self.dataloader)
        for i, (src, tgt) in enumerate(eph):
            loss = self._step(src, tgt)

            if first_loss is None:
                first_loss = loss.item()

            if self.cfg.max_iter > 0:
                if i >= self.cfg.max_iter:
                    break

            if (i + 1) % self.cfg.update_desc_every_n_steps == 0:
                loss_percent = (loss.item() / first_loss) * 100
                eph.set_description(f"Loss: {loss.item():.4f} | {loss_percent:.2f}%")

            if (i + 1) % self.cfg.save_every_n_steps == 0:
                print(f"Saving model at epoch {i + 1}")
                self.save_model()

        print("Saving final model")
        self.save_model()

lm_config = Config(
    dim = 256,
    hidden= 1024,
    dss_dim = 256,
    dss_hidden = 128,
    vocab_size = 32002,
    layers = 4,
    dropout = 0.1
)

model = KAN_GSS_LM(lm_config)

import tokenmonster
from datasets import load_dataset

stories = load_dataset("roneneldan/TinyStories", split="train")[0:1000]
stories = ["<s>"+i for i in stories['text']]
tokenizer = tokenmonster.load("englishcode-32000-consistent-nocapcode-v1")

tokenizer.add_special_token("<s>")
tokenizer.add_special_token("</s>")

stories = tokenizer.tokenize("</s>\n".join(stories)).tolist()

train_config = TrainConfig(
    lr = 0.00005,
    block_size = 512,
    batch_size = 8,
    save_every_n_steps = 250,
    update_desc_every_n_steps = 25,
    max_iter = 1000
)

data = Data(stories, train_config)
train = Train(model, data, train_config)

train.train()

tensor = torch.tensor([tokenizer.tokenize("<s>").tolist()], dtype=torch.int64).to(train.cfg.device)

for i in train.model.generate(tensor, 100, temperature=0.65):
    print(tokenizer.decode(i), end="", flush=True)
