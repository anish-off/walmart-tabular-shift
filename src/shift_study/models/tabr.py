"""TabR-S (ICLR 2024): feed-forward encoder + differentiable k-NN retrieval.
For each query x: encode h=E(x), key k_x=K(h); retrieve top-m training candidates
by -||k_x - k_i||^2; context value v_i = Wy(y_i) + T(k_i - k_x); output =
head(h + sum_i softmax(sim)_i * v_i).

M5-scale concessions from the plan: candidates = 500K-row training subsample,
candidate keys frozen after `context_freeze_epoch` epochs, chunked top-m search.
Self-retrieval is masked during training (a training row must not retrieve its
own label).

Gradient note: chunked_topm is used only for top-m *index selection* (run under
no_grad for speed). Attention logits (sims_live) are recomputed from the selected
candidate keys and the live k_x WITH gradient, so the encoder and key network
are fully trained through the retrieval attention path."""
import numpy as np
import torch
import torch.nn as nn

from .base import TabularModel
from .torch_common import EarlyStopper, Preprocessor, emb_dim


class TabRNet(nn.Module):
    def __init__(self, n_num: int, cards: list[int], d: int, dropout: float):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim(c)) for c in cards])
        d_in = n_num + sum(e.embedding_dim for e in self.embs)
        self.enc = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d, d))
        self.key = nn.Linear(d, d)
        self.label_emb = nn.Linear(1, d)
        self.t = nn.Sequential(nn.Linear(d, d), nn.ReLU(),
                               nn.Dropout(dropout), nn.Linear(d, d))
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d, 1))
        self.scale = d ** -0.5

    def featurize(self, x_num, x_cat):
        parts = [x_num] + [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        return torch.cat(parts, dim=1)

    def encode(self, x_num, x_cat):
        return self.enc(self.featurize(x_num, x_cat))

    def forward_with_context(self, h, cand_keys_sel, cand_y_sel, sims):
        """h: (B,d); cand_keys_sel: (B,m,d); cand_y_sel: (B,m); sims: (B,m)."""
        kx = self.key(h)                                   # (B, d)
        alpha = torch.softmax(sims * self.scale, dim=1)    # (B, m)
        v = self.label_emb(cand_y_sel.unsqueeze(-1)) \
            + self.t(cand_keys_sel - kx.unsqueeze(1))      # (B, m, d)
        ctx = (alpha.unsqueeze(-1) * v).sum(dim=1)         # (B, d)
        return self.head(h + ctx).squeeze(-1)


def chunked_topm(kx: torch.Tensor, cand_keys: torch.Tensor, m: int,
                 chunk: int, exclude: torch.Tensor | None = None):
    """Top-m candidates by -squared-distance, scanning candidates in chunks.
    kx: (B,d), cand_keys: (N,d), exclude: (B,) candidate indices to mask (self).
    Returns (sims (B,m), idx (B,m))."""
    best_s, best_i = None, None
    n = len(cand_keys)
    for s in range(0, n, chunk):
        block = cand_keys[s:s + chunk]                     # (C, d)
        sims = -torch.cdist(kx, block).pow(2)              # (B, C)
        if exclude is not None:
            # Identify rows whose excluded index falls in this block
            in_block = (exclude >= s) & (exclude < s + len(block))
            if in_block.any():
                # Use explicit row indices (safer than boolean+integer mixed indexing)
                r = torch.nonzero(in_block).squeeze(1)
                sims[r, exclude[r] - s] = float("-inf")
        idx = torch.arange(s, s + len(block), device=kx.device).expand(len(kx), -1)
        if best_s is None:
            best_s, best_i = sims, idx
        else:
            best_s = torch.cat([best_s, sims], dim=1)
            best_i = torch.cat([best_i, idx], dim=1)
        if best_s.shape[1] > 4 * m:                        # keep memory bounded
            top = best_s.topk(min(m, best_s.shape[1]), dim=1)
            best_i = best_i.gather(1, top.indices)
            best_s = top.values
    top = best_s.topk(min(m, best_s.shape[1]), dim=1)
    return top.values, best_i.gather(1, top.indices)


class TabRModel(TabularModel):
    name = "tabr"

    def fit(self, X, y, X_val, y_val):
        p = self.params
        device = p.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # candidate subsample (plan: 500K max)
        n_sub = min(int(p.get("train_subsample", 500000)), len(X))
        sub = rng.choice(len(X), size=n_sub, replace=False)
        X, y = X.iloc[sub].reset_index(drop=True), np.asarray(y)[sub]

        self.prep = Preprocessor().fit(X)
        xn, xc = self.prep.transform(X, "cpu")
        # Normalize labels so label_emb operates on O(1) scale values, not raw
        # target magnitudes that would overwhelm the feature representation h.
        y_arr = y.astype(np.float32)
        self.y_mean = float(y_arr.mean())
        self.y_std = float(y_arr.std()) + 1e-6
        y_t = torch.tensor((y_arr - self.y_mean) / self.y_std)
        vxn, vxc = self.prep.transform(X_val, "cpu")
        vy_norm = torch.tensor(
            (np.asarray(y_val, dtype=np.float32) - self.y_mean) / self.y_std)

        d = int(p.get("d_main", 96))
        self.m = int(p.get("context_size", 96))
        self.chunk = int(p.get("candidate_chunk", 65536))
        self.net = TabRNet(xn.shape[1], self.prep.cards, d,
                           float(p.get("dropout", 0.1))).to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(p.get("lr", 5e-4)),
                                weight_decay=float(p.get("weight_decay", 1e-5)))
        stopper = EarlyStopper(int(p.get("patience", 10)))
        bs = int(p.get("batch_size", 1024))
        freeze_at = int(p.get("context_freeze_epoch", 4))
        n = len(xn)
        frozen_keys = None

        # Hoist label tensor to device once; index per batch with y_dev[idx].
        y_dev = y_t.to(device)

        for epoch in range(int(p.get("max_epochs", 100))):
            # candidate keys: recomputed each epoch until frozen
            if frozen_keys is None or epoch < freeze_at:
                with torch.no_grad():
                    keys = self._all_keys(xn, xc, device, bs)
                if epoch == freeze_at - 1 or freeze_at == 0:
                    frozen_keys = keys
            cand_keys = frozen_keys if frozen_keys is not None else keys

            self.net.train()
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                xb_n, xb_c = xn[b].to(device), xc[b].to(device)
                yb = y_dev[b]
                h = self.net.encode(xb_n, xb_c)
                kx = self.net.key(h)
                # Use no_grad only for top-m index selection; recompute logits
                # with gradient so the key network is trained via the attention path.
                with torch.no_grad():
                    _, idx = chunked_topm(kx.detach(), cand_keys, self.m,
                                         self.chunk, exclude=b.to(device))
                sel_keys = cand_keys[idx]                  # (B, m, d), no grad
                sel_y = y_dev[idx]
                sims_live = -((sel_keys - kx.unsqueeze(1)) ** 2).sum(-1)  # grad via kx
                pred = self.net.forward_with_context(h, sel_keys, sel_y, sims_live)
                loss = (pred - yb).abs().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            # validation (compare in normalized scale)
            self.net.eval()
            with torch.no_grad():
                cand_eval = frozen_keys if frozen_keys is not None else \
                    self._all_keys(xn, xc, device, bs)
                val_pred = self._predict_from(vxn, vxc, cand_eval, y_t, device, bs)
                val_mae = (val_pred.cpu() - vy_norm).abs().mean().item()
            if stopper.step(val_mae, self.net):
                break
        stopper.restore(self.net)
        # Val MAE was measured against frozen_keys, so deploy those same keys for
        # inference; only recompute via _all_keys when training never reached the
        # freeze epoch (frozen_keys is None).
        if frozen_keys is not None:
            self.cand_keys = frozen_keys
        else:
            with torch.no_grad():
                self.cand_keys = self._all_keys(xn, xc, device, bs)
        self.cand_y = y_t  # normalized training labels
        self.device, self.bs = device, bs
        self.model = self.net
        return self

    def _all_keys(self, xn, xc, device, bs):
        keys = []
        for s in range(0, len(xn), bs):
            h = self.net.encode(xn[s:s + bs].to(device), xc[s:s + bs].to(device))
            keys.append(self.net.key(h))
        return torch.cat(keys)

    def _predict_from(self, xn, xc, cand_keys, cand_y, device, bs):
        if len(xn) == 0:
            return torch.zeros(0)
        out = []
        cand_y_dev = cand_y.to(device)
        for s in range(0, len(xn), bs):
            h = self.net.encode(xn[s:s + bs].to(device), xc[s:s + bs].to(device))
            kx = self.net.key(h)
            # Recompute sims on selected candidates with the same formula as training
            # (-Σ(k_i - k_x)^2 == -cdist^2), ensuring train/eval consistency.
            _, idx = chunked_topm(kx, cand_keys, self.m, self.chunk)
            sel_keys = cand_keys[idx]
            sims_live = -((sel_keys - kx.unsqueeze(1)) ** 2).sum(-1)
            out.append(self.net.forward_with_context(
                h, sel_keys, cand_y_dev[idx], sims_live))
        return torch.cat(out)

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict()")
        self.net.eval()
        xn, xc = self.prep.transform(X, "cpu")
        pred_norm = self._predict_from(xn, xc, self.cand_keys, self.cand_y,
                                       self.device, self.bs)
        return (pred_norm.cpu().numpy() * self.y_std + self.y_mean)
