"""TabM (ICLR 2025): parameter-efficient ensemble of k MLPs sharing one backbone.
BatchEnsemble-style: shared weight matrix W per layer plus per-member rank-1
adapters (r_i input scale, s_i output scale, b_i bias), random-sign initialized.
Prediction = mean over the k members. Loss = mean member MAE."""
import numpy as np
import torch
import torch.nn as nn

from .base import TabularModel
from .torch_common import EarlyStopper, Preprocessor, emb_dim


def _random_sign(*shape):
    return torch.where(torch.rand(*shape) < 0.5, -1.0, 1.0)


class BatchEnsembleLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, k: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.r = nn.Parameter(_random_sign(k, d_in))
        self.s = nn.Parameter(_random_sign(k, d_out))
        self.bias = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x):  # x: (B, k, d_in)
        return ((x * self.r) @ self.weight) * self.s + self.bias


class TabMNet(nn.Module):
    def __init__(self, n_num: int, cards: list[int], d_main: int, k: int,
                 n_blocks: int, dropout: float):
        super().__init__()
        self.k = k
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim(c)) for c in cards])
        d_in = n_num + sum(e.embedding_dim for e in self.embs)
        self.blocks = nn.ModuleList()
        d = d_in
        for _ in range(n_blocks):
            self.blocks.append(BatchEnsembleLinear(d, d_main, k))
            d = d_main
        self.drop = nn.Dropout(dropout)
        self.head = BatchEnsembleLinear(d_main, 1, k)

    def forward(self, x_num, x_cat):  # returns (B, k)
        parts = [x_num] + [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        z = torch.cat(parts, dim=1)
        z = z.unsqueeze(1).expand(-1, self.k, -1)
        for blk in self.blocks:
            z = self.drop(torch.relu(blk(z)))
        return self.head(z).squeeze(-1)


class TabMModel(TabularModel):
    name = "tabm"

    def fit(self, X, y, X_val, y_val):
        p = self.params
        device = p.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

        self.prep = Preprocessor().fit(X)
        # Hoist all training data to device once; eliminates 1.5M per-batch CPU→GPU copies.
        xn, xc = self.prep.transform(X, device)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32), device=device)
        vxn, vxc = self.prep.transform(X_val, device)
        vy = torch.tensor(np.asarray(y_val, dtype=np.float32), device=device)

        self.net = TabMNet(
            n_num=xn.shape[1], cards=self.prep.cards,
            d_main=int(p.get("d_main", 512)), k=int(p.get("k", 8)),
            n_blocks=int(p.get("n_blocks", 3)), dropout=float(p.get("dropout", 0.0)),
        ).to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(p.get("lr", 1e-3)),
                                weight_decay=float(p.get("weight_decay", 1e-5)))
        stopper = EarlyStopper(int(p.get("patience", 20)))
        bs = int(p.get("batch_size", 4096))
        max_epochs = int(p.get("max_epochs", 200))
        log_every = int(p.get("log_every", 10))

        n = len(xn)
        for epoch in range(max_epochs):
            self.net.train()
            perm = torch.randperm(n, device=device)
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                pred = self.net(xn[b], xc[b])              # (B, k)
                loss = (pred - y_t[b].unsqueeze(1)).abs().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val_pred = self._predict_tensor(vxn, vxc, bs)
                val_mae = (val_pred - vy).abs().mean().item()
            if (epoch + 1) % log_every == 0:
                print(f"  [tabm] epoch {epoch+1:3d}/{max_epochs}  val_mae={val_mae:.4f}",
                      flush=True)
            if stopper.step(val_mae, self.net):
                print(f"  [tabm] early stop at epoch {epoch+1}  val_mae={val_mae:.4f}",
                      flush=True)
                break
        stopper.restore(self.net)
        self.device, self.bs = device, bs
        self.model = self.net  # satisfy base-class fitted marker
        return self

    def _predict_tensor(self, xn, xc, bs, device=None):
        if len(xn) == 0:
            return torch.zeros(0)
        out = []
        for s in range(0, len(xn), bs):
            bn, bc = xn[s:s + bs], xc[s:s + bs]
            if device is not None:
                bn, bc = bn.to(device), bc.to(device)
            out.append(self.net(bn, bc).mean(dim=1))
        return torch.cat(out)

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict()")
        self.net.eval()
        # Transform to CPU; _predict_tensor moves each batch to device, matching
        # the training loop's memory pattern and avoiding putting the full test
        # set on GPU at once.
        xn, xc = self.prep.transform(X, "cpu")
        return self._predict_tensor(xn, xc, self.bs, device=self.device).cpu().numpy()

    def save(self, path) -> None:
        torch.save({
            "state_dict": self.net.state_dict(),
            "prep": self.prep,
            "config": self.config,
            "seed": self.seed,
        }, path)

    @classmethod
    def load(cls, path) -> "TabMModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(ckpt["config"], seed=ckpt["seed"])
        obj.prep = ckpt["prep"]
        p = ckpt["config"].get("params", {})
        obj.net = TabMNet(
            n_num=len(obj.prep.num_cols), cards=obj.prep.cards,
            d_main=int(p.get("d_main", 512)), k=int(p.get("k", 8)),
            n_blocks=int(p.get("n_blocks", 3)), dropout=float(p.get("dropout", 0.0)),
        )
        sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["state_dict"].items()}
        obj.net.load_state_dict(sd)
        obj.net.eval()
        obj.model = obj.net
        device = p.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        obj.net.to(device)
        obj.device = device
        obj.bs = int(p.get("batch_size", 4096))
        return obj
