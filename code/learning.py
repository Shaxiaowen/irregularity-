"""Data-driven sensing of the airflow condition with leave-one-group-out validation."""
import numpy as np
import shap
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def elastic_net():
    return make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.1, .5, .9], cv=KFold(5, shuffle=True, random_state=0), max_iter=20000))


def boosted_trees():
    return xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=3, reg_lambda=2.0, random_state=0)


def cross_validated_predictions(model_factory, X, y, groups):
    """Out-of-group predictions: each group (collection day) is held out once."""
    pred = np.zeros_like(y, dtype=float)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = model_factory().fit(X[tr], y[tr]).predict(X[te])
    return pred


def bootstrap_r2_difference(y, pred_a, pred_b, groups, n_boot=2000, seed=0):
    """95% interval of R2(a) - R2(b) from a group-block bootstrap."""
    rng = np.random.default_rng(seed); g = np.unique(groups); d = []
    for _ in range(n_boot):
        sel = np.concatenate([np.flatnonzero(groups == k) for k in rng.choice(g, len(g))])
        d.append(r2_score(y[sel], pred_a[sel]) - r2_score(y[sel], pred_b[sel]))
    return np.percentile(d, [2.5, 97.5])


def shap_importance(X, y, feature_names):
    model = boosted_trees().fit(X, y)
    values = shap.TreeExplainer(model).shap_values(X)
    return dict(sorted(zip(feature_names, np.abs(values).mean(0)), key=lambda kv: -kv[1]))


class SignalCNN(nn.Module):
    """1D-CNN on multichannel residual sequences, with operating conditions appended before the head."""

    def __init__(self, n_channels, n_conditions):
        super().__init__()
        block = lambda i, o, k: [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()]
        self.features = nn.Sequential(*block(n_channels, 16, 7), nn.MaxPool1d(2), *block(16, 32, 5), nn.MaxPool1d(2),
                                      *block(32, 64, 5), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(64 + n_conditions, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, seq, cond):
        return self.head(torch.cat([self.features(seq).squeeze(-1), cond], 1)).squeeze(-1)


def train_cnn(seqs, cond, y, crop=400, epochs=60, batch=32, seed=0):
    """Train on random crops; seqs is a list of (channels, length) arrays, cond is standardized."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = SignalCNN(seqs[0].shape[0], cond.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    ym, ys = y.mean(), y.std()
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(len(seqs)), max(1, len(seqs) // batch)):
            s = torch.tensor(np.stack([seqs[i][:, (o := rng.integers(0, seqs[i].shape[1] - crop + 1)):o + crop] for i in b]), dtype=torch.float32)
            loss = nn.functional.mse_loss(net(s, torch.tensor(cond[b], dtype=torch.float32)),
                                          torch.tensor((y[b] - ym) / ys, dtype=torch.float32))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, ym, ys
