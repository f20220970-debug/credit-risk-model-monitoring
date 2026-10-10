"""Weight-of-evidence (WoE) binning and information value (IV).

For each feature we split values into bins, then for every bin:
    WoE = ln( share of all goods in the bin / share of all bads in the bin )
A positive WoE means the bin is safer than average; negative means riskier.
    IV  = sum over bins of (share of goods - share of bads) * WoE
IV measures how well the whole feature separates goods from bads.
Rule of thumb: < 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, > 0.3 strong.
"""
import numpy as np
import pandas as pd

MISSING = "Missing"


class WoEBinner:
    def __init__(self, n_bins=10, min_category_share=0.01, smoothing=0.5):
        self.n_bins = n_bins
        self.min_category_share = min_category_share
        self.smoothing = smoothing   # avoids ln(0) when a bin has no goods or no bads
        self.edges = {}       # numeric feature -> bin edges
        self.categories = {}  # categorical feature -> categories kept
        self.woe = {}         # feature -> {bin label: WoE}
        self.iv = {}          # feature -> IV

    # ---- binning ----------------------------------------------------------
    def _bin_numeric(self, x: pd.Series, edges) -> pd.Series:
        labels = pd.cut(x, bins=edges, include_lowest=True).astype("string")
        return labels.fillna(MISSING).astype(str)

    def _bin_categorical(self, x: pd.Series, keep) -> pd.Series:
        x = x.astype("string").fillna(MISSING)
        return x.where(x.isin(keep) | (x == MISSING), "Other").astype(str)

    def bin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the bin label of every value (used for WoE and for CSI)."""
        out = {}
        for f, edges in self.edges.items():
            out[f] = self._bin_numeric(df[f], edges)
        for f, keep in self.categories.items():
            out[f] = self._bin_categorical(df[f], keep)
        return pd.DataFrame(out, index=df.index)

    # ---- fitting ----------------------------------------------------------
    def fit(self, df, y, numeric, categorical):
        y = pd.Series(np.asarray(y), index=df.index)
        for f in numeric:
            q = np.nanquantile(df[f].astype(float), np.linspace(0, 1, self.n_bins + 1))
            edges = np.unique(q)
            edges[0], edges[-1] = -np.inf, np.inf
            self.edges[f] = edges
        for f in categorical:
            share = df[f].astype("string").fillna(MISSING).value_counts(normalize=True)
            self.categories[f] = [c for c, s in share.items()
                                  if s >= self.min_category_share and c != MISSING]
        binned = self.bin(df)
        for f in binned.columns:
            self.woe[f], self.iv[f] = self._woe_iv(binned[f], y)
        return self

    def _woe_iv(self, bins: pd.Series, y: pd.Series):
        t = pd.crosstab(bins, y)
        t = t.reindex(columns=[0, 1], fill_value=0)
        k = len(t)
        good = (t[0] + self.smoothing) / (t[0].sum() + self.smoothing * k)
        bad = (t[1] + self.smoothing) / (t[1].sum() + self.smoothing * k)
        woe = np.log(good / bad)
        iv = float(((good - bad) * woe).sum())
        return woe.to_dict(), iv

    # ---- applying ---------------------------------------------------------
    def transform(self, df: pd.DataFrame, features=None) -> pd.DataFrame:
        """Replace every value by the WoE of its bin. Unseen bins get WoE 0 (neutral)."""
        binned = self.bin(df)
        features = features or list(binned.columns)
        return pd.DataFrame({f: binned[f].map(self.woe[f]).fillna(0.0).astype(float)
                             for f in features}, index=df.index)

    def bin_table(self, df, y) -> pd.DataFrame:
        """Long table: feature, bin, loans, default rate, WoE (for the report and Tableau)."""
        binned = self.bin(df)
        rows = []
        y = pd.Series(np.asarray(y), index=df.index)
        for f in binned.columns:
            g = y.groupby(binned[f]).agg(["size", "mean"])
            for b, r in g.iterrows():
                rows.append({"feature": f, "bin": b, "loans": int(r["size"]),
                             "default_rate": round(r["mean"], 4),
                             "woe": round(self.woe[f].get(b, 0.0), 4)})
        return pd.DataFrame(rows)

    def iv_table(self) -> pd.DataFrame:
        t = pd.DataFrame({"feature": list(self.iv), "iv": list(self.iv.values())})
        t["strength"] = pd.cut(t["iv"], [-np.inf, 0.02, 0.1, 0.3, np.inf],
                               labels=["useless", "weak", "medium", "strong"])
        return t.sort_values("iv", ascending=False).reset_index(drop=True)
