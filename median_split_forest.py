import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RF_FILE = os.path.join(os.path.dirname(BASE_DIR), "Data_NEU", "factor", "rf_factor.csv")
RF_START_YEAR = 1964
RF_START_MONTH = 1

def to_datetime(values: Union[pd.Series, np.ndarray]) -> pd.Series:
    s = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    dt = pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce")
    return dt


def month_keys(values: Union[pd.Series, np.ndarray]) -> np.ndarray:
    dt = to_datetime(values)
    return dt.dt.to_period("M").astype(str).to_numpy()


def get_rf_series(
    rf_path: Optional[str] = None,
    ) -> np.ndarray:
    path = rf_path or DEFAULT_RF_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(f"rf factor file not found: {path}")
    rf_df = pd.read_csv(path, header=None)
    if rf_df.shape[1] != 1:
        raise ValueError(f"Expected single rf column, found {rf_df.shape[1]} in {path}")
    rf_series = pd.to_numeric(rf_df.iloc[:, 0], errors="coerce")
    if rf_series.isna().any():
        bad = rf_df[rf_series.isna()].head(5)
        raise ValueError(f"Non-numeric rf entries detected in {path}:\n{bad}")
    rf_vals = rf_series.to_numpy(dtype=float) / 100.0
    return rf_vals

@dataclass
class Node:
    node_id: int
    printed_path_code: str              
    depth: int
    is_leaf: bool
    n_obs: int
    feature: Optional[int] = None
    feature_name: Optional[str] = None
    threshold: Optional[float] = None
    left_id: Optional[int] = None
    right_id: Optional[int] = None
    idx: Optional[np.ndarray] = None 
    idx_test : Optional[np.ndarray] = None 
    monthly_excess_returns: Optional[List[float]] = None


class MedianSplitTree:
    """
    Binary tree with median splits.
    - At each node: choose random subset of features (max_features), then pick the feature that maximizes the Sharpe ratio of monthly excess returns between left and right child nodes (median split)
    - Stores the monthly excess returns for each node (if computable from the data and rf series), which can be exported later. 
    """

    def __init__(
        self,
        max_depth,
        max_features,
        min_samples_leaf,
        random_state,
        max_tries_per_node,
        rf_series: np.ndarray,
        threshold_candidates: Optional[List[float]] = None,
    ):
        self.max_depth = int(max_depth)
        self.max_features = max_features
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state
        self.rf_series = rf_series
        self.max_tries_per_node = int(max_tries_per_node)
        self.threshold_candidates = threshold_candidates
        self.rng_ = np.random.RandomState(random_state)
        self.nodes_: List[Node] = []
        self._next_node_id = 0
        self.n_features_in_ = None

    def _resolve_max_features(self, p: int) -> int:
        mf = self.max_features
        if mf is None:
            return p
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(p)))
            if mf == "log2":
                return max(1, int(np.log2(p)))
            raise ValueError(f"Unknown max_features string: {mf}")
        if isinstance(mf, float):
            return max(1, int(np.ceil(mf * p)))
        return max(1, min(p, int(mf)))
    
    def count_sampled_months(self, month_keys: np.ndarray, sampled_months: np.ndarray) -> Dict[str, int]:
        """Return month -> sample multiplicity for the months present in month_keys.

        Ensures keys align with _build_month_to_rf: string periods, NaT filtered, only
        parsable months retained. Months not drawn get a count of 0.
        """
        if sampled_months is None:
            return {}

        # normalize inputs and bail out early if nothing valid
        valid = month_keys[month_keys != "NaT"]
        if valid.size == 0:
            return {}

        sampled = sampled_months[sampled_months != "NaT"]
        if sampled.size == 0:
            return {mk: 0 for mk in np.unique(valid)}

        uniq_sampled, counts_sampled = np.unique(sampled, return_counts=True)
        sampled_map = dict(zip(uniq_sampled, counts_sampled.astype(int)))

        uniq_valid = np.unique(valid)
        return {mk: int(sampled_map.get(mk, 0)) for mk in uniq_valid}

    def _month_index_from_start(self, period_str: str) -> Optional[int]:
        try:
            p = pd.Period(period_str, freq="M")
        except Exception:
            return None
        return (p.year - (RF_START_YEAR - 1)) * 12 + (p.month - RF_START_MONTH)

    def _build_month_to_rf(self, month_keys: np.ndarray, rf_series: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
        if rf_series is None:
            return None
        valid = month_keys[month_keys != "NaT"]
        uniq_months = np.unique(valid)
        tmp: Dict[str, float] = {}
        for m in uniq_months:
            idx_rf = self._month_index_from_start(m)
            if idx_rf is None:
                continue
            if 0 <= idx_rf < rf_series.size:
                tmp[m] = float(rf_series[idx_rf])
        return tmp if tmp else None

    def _monthly_excess_returns(
        self,
        vals: np.ndarray,
        weights: Optional[np.ndarray],
        month_keys: np.ndarray,
        monthly_rf: Optional[Dict[str, float]],
        sampled_month_counts: Optional[Dict[str, int]] = None,
    ) -> Optional[np.ndarray]:
        if vals.size < 1 or monthly_rf is None:
            return None

        # filter out invalid months once
        valid_mask = month_keys != "NaT"
        if not np.any(valid_mask):
            return None

        vals_valid = vals[valid_mask]
        months_valid = month_keys[valid_mask]

        if weights is not None:
            if weights.size != vals.size:
                weights_valid = None
            else:
                weights_valid = weights[valid_mask]
        else:
            weights_valid = None

        uniq_months, inv_codes = np.unique(months_valid, return_inverse=True)
        n_months = uniq_months.size
        if n_months == 0:
            return None

        counts = np.bincount(inv_codes)
        sum_vals = np.bincount(inv_codes, weights=vals_valid)
        mean_unweighted = sum_vals / counts

        if weights_valid is not None:
            sum_w = np.bincount(inv_codes, weights=weights_valid)
            sum_wret = np.bincount(inv_codes, weights=weights_valid * vals_valid)
        else:
            sum_w = None
            sum_wret = None

        rets: List[float] = []

        for i, m_key in enumerate(uniq_months):
            rf_val = monthly_rf.get(m_key)
            if rf_val is None:
                return None

            if sum_w is not None:
                wsum = sum_w[i]
                if wsum > 0:
                    r = float(sum_wret[i] / wsum)
                else:
                    r = float(mean_unweighted[i])
            else:
                r = float(mean_unweighted[i])

            multiplicity = 1
            if sampled_month_counts is not None:
                multiplicity = int(sampled_month_counts.get(m_key, 1))
                if multiplicity < 1:
                    multiplicity = 0

            if multiplicity == 0:
                continue

            rets.extend([r - rf_val] * multiplicity)

        rets_arr = np.asarray(rets, dtype=float)
        return rets_arr if rets_arr.size >= 1 else None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, idx: np.ndarray, sampled_months: np.ndarray, feature_names: List[str]) -> "MedianSplitTree":
        
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows.")

        self.n_features_in_ = len(feature_names)
        self.nodes_ = []
        self._next_node_id = 0

        self._build_node(
            X=X,
            y=y,
            idx=idx,
            sampled_months=sampled_months,
            feature_names=feature_names,
            depth=0,
            printed_path_code="1",
            rf_series=self.rf_series,
            parent_node_id=None,
            is_left_child=None,
            feature_seq = None,
            path_code="1"
        )
        return self

    def _new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def _build_node(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        idx: np.ndarray,
        sampled_months: np.ndarray,
        feature_names: List[str],
        depth: int,
        printed_path_code: str,
        rf_series: np.ndarray,
        parent_node_id: Optional[int],
        is_left_child: Optional[bool],
        feature_seq: Optional[list] = None,
        path_code: Optional[str] = None
    ) -> int:
        
        if feature_seq is None:
            feature_seq = []
        
        X_sparse = X[feature_names].to_numpy()
        X_sparse_node = X_sparse[idx, :]
        node_id = self._new_node_id()

        size_all = X["size"].to_numpy()
        size_node = size_all[idx]
        date_all = X["date"].to_numpy()
        date_node = date_all[idx]
        y_node = y[idx]

        month_keys_node = month_keys(date_node)
        monthly_rf = self._build_month_to_rf(month_keys_node, rf_series)
        sampled_month_counts = self.count_sampled_months(month_keys_node, sampled_months)

        # precompute per-month index lists for reuse in split evaluation
        unique_months_node = np.unique(month_keys_node)
        unique_months_node = unique_months_node[unique_months_node != "NaT"]
        month_indices = [np.where(month_keys_node == mk)[0] for mk in unique_months_node]

        # store monthly excess returns for this node (if available)
        node_monthly = self._monthly_excess_returns(y_node, size_node, month_keys_node, monthly_rf, sampled_month_counts)

        node = Node(
            node_id=node_id,
            printed_path_code=printed_path_code,
            depth=depth,
            is_leaf=True,
            n_obs=int(idx.size),
            idx=idx.copy(),
            monthly_excess_returns=[node_monthly[i] for i in range(node_monthly.size)] if node_monthly is not None else None
        )
        self.nodes_.append(node)

        # attach to parent
        if parent_node_id is not None:
            parent = self.nodes_[parent_node_id]
            if is_left_child:
                parent.left_id = node_id
            else:
                parent.right_id = node_id

        # stopping rules
        if depth >= self.max_depth:
            return node_id
        if idx.size < 2 * self.min_samples_leaf:
            print(f"Stopping rule min_samples_leaf reached at node_id {node_id} with {idx.size} observations, depth {depth}.")
            return node_id

        p = self.n_features_in_
        m = self._resolve_max_features(p)

        best_score = -np.inf
        best_split = None  # (feature_index, left_mask, right_mask, threshold)

        feature_iter = self.rng_.choice(p, size=m, replace=False)

        for j in feature_iter:
            xj = X_sparse_node[:, j]
            if xj.size == 0:
                continue
            
            threshold_global = np.median(xj)
            if threshold_global is None or not np.isfinite(threshold_global):
                continue

            left_mask = xj <= threshold_global
            n_left = int(left_mask.sum())
            n_right = int(xj.size - n_left)

            # global min leaf check
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue

            # per-month min leaf check using precomputed indices
            per_month_ok = True
            for m_idx, idx_m in enumerate(month_indices):
                l_count = int(left_mask[idx_m].sum())
                r_count = int(idx_m.size - l_count)
                if l_count < self.min_samples_leaf or r_count < self.min_samples_leaf:
                    per_month_ok = False
                    break

            if not per_month_ok:
                continue

            right_mask = np.logical_not(left_mask)

            left_idx = idx[left_mask]
            right_idx = idx[right_mask]

            left_w = size_node[left_mask]
            right_w = size_node[right_mask]

            left_ret = self._monthly_excess_returns(y[left_idx], left_w, month_keys_node[left_mask], monthly_rf, sampled_month_counts)
            right_ret = self._monthly_excess_returns(y[right_idx], right_w, month_keys_node[right_mask], monthly_rf, sampled_month_counts)

            if left_ret is None or right_ret is None:
                continue
            if left_ret.shape[0] != right_ret.shape[0]:
                continue

            monthly_spread_return = right_ret - left_ret
            if monthly_spread_return.size < 2:
                continue

            spread_std = np.std(monthly_spread_return)
            if spread_std <= 0 or not np.isfinite(spread_std):
                continue

            score = abs(np.mean(monthly_spread_return) / spread_std)

            if score > best_score:
                best_score = score
                best_split = (j, left_mask.copy(), right_mask.copy(), threshold_global)

        if best_split is None:
            return node_id

        j, left_mask, right_mask, threshold_global = best_split

        node.is_leaf = False
        node.feature = j
        node.feature_name = feature_names[j]
        node.threshold = threshold_global

        new_feature_seq = feature_seq + [j]
        feature_seq_str = "".join(str(int(f)) for f in new_feature_seq)

        left_idx = idx[left_mask]
        right_idx = idx[right_mask]

        left_lr = path_code + "1"
        right_lr = path_code + "2"

        left_path_code = f"{feature_seq_str}.{left_lr}"
        right_path_code = f"{feature_seq_str}.{right_lr}"

        self._build_node(X, y, left_idx, sampled_months, feature_names, depth + 1, left_path_code, rf_series, node_id, True, new_feature_seq, left_lr)
        self._build_node(X, y, right_idx, sampled_months, feature_names, depth + 1, right_path_code, rf_series, node_id, False, new_feature_seq, right_lr)
        
        return node_id

    def export_tables(
            self,
            tree_id: int,
            feature_names: Optional[List[str]] = None,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """Export per-node metadata plus a wide table of monthly excess returns.
            Returns
            -------
            nodes_df : DataFrame
                One row per node with metadata (idx dropped).
            excess_df : DataFrame
                Wide format: each column is a node's path_code, each row a monthly
                excess return for that node (padded with NaN to the max length).
            """

            nodes_df = pd.DataFrame([n.__dict__ for n in self.nodes_])
            nodes_df.insert(0, "tree_id", tree_id)

            for col in ["node_id", "is_leaf", "idx", "monthly_excess_returns"]:
                if col in nodes_df.columns:
                    nodes_df = nodes_df.drop(columns=[col])

            excess_cols: Dict[str, List[float]] = {}
            max_len = 0

            for n in self.nodes_:
                if n.monthly_excess_returns is None:
                    continue
                vals = list(n.monthly_excess_returns)
                excess_cols[n.printed_path_code] = vals
                if len(vals) > max_len:
                    max_len = len(vals)

            if excess_cols:
                excess_df = pd.DataFrame(excess_cols)
            else:
                excess_df = pd.DataFrame()

            return nodes_df, excess_df
    
    def result(self, X_arr: pd.DataFrame, feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Route rows through stored splits (feature + threshold) and return path -> indices for leaf nodes only.

        Side-effect: writes the test row indices into each leaf node's "idx_test" attribute.
        """
        if not self.nodes_:
            raise ValueError("Tree not fitted.")

        X_feat = X_arr[feature_names].to_numpy()

        results: Dict[str, List[int]] = {}
        path_to_node: Dict[str, Node] = {}

        for row_idx, row in enumerate(X_feat):

            node = self.nodes_[0]

            while True:
                # stop if leaf or invalid split info
                if node.is_leaf or node.feature is None:
                    break

                thr = node.threshold
                feat = node.feature
                if thr is None or not np.isfinite(thr) or feat is None:
                    break

                go_left = row[feat] <= thr
                next_id = node.left_id if go_left else node.right_id
                if next_id is None:
                    break

                node = self.nodes_[next_id]

            path = node.printed_path_code
            if path not in results:
                results[path] = []
                path_to_node[path] = node
            results[path].append(row_idx)

        finalized = {k: np.asarray(v, dtype=int) for k, v in results.items()}

        # write idx_test back to leaf nodes only
        for path, idxs in finalized.items():
            n = path_to_node.get(path)
            if n is not None and n.is_leaf:
                n.idx_test = idxs

        return finalized

    def _traverse_to_leaf(self, x_row: np.ndarray) -> Node:
        """Traverse the tree for one sample vector and return the reached leaf node."""
        if not self.nodes_:
            raise ValueError("Tree not fitted.")

        node = self.nodes_[0]  # root
        while not node.is_leaf:
            if node.feature is None or node.threshold is None or not np.isfinite(node.threshold):
                break
            go_left = x_row[node.feature] <= node.threshold
            next_id = node.left_id if go_left else node.right_id
            if next_id is None:
                break
            node = self.nodes_[next_id]
        return node

    def get_leaf_neighbors(
        self,
        X_new: pd.DataFrame,
        feature_names: List[str],
        row_id: int
    ) -> List[Tuple[int, np.ndarray]]:
        """
    Route only the provided ``row_id`` through the tree and return its leaf neighbors.

        Parameters
        ----------
        X_new : DataFrame
            Test data with the same columns used during fit.
        feature_names : list[str]
            Order of feature columns (must match the order used in fit).
        row_id : int
            Row index to route through the tree.

        Returns
        -------
        List[Tuple[int, np.ndarray]]
            For the routed row, returns a tuple (row_id, idxs) where `idxs` are training indices of the reached leaf.
        """
        if row_id is None:
            raise ValueError("row_id must be provided to get_leaf_neighbors")

        X_arr = X_new[feature_names].to_numpy()
        neighbors: List[Tuple[int, np.ndarray]] = []

        row = X_arr[row_id]
        leaf = self._traverse_to_leaf(row)
        idxs = leaf.idx_test if leaf.idx_test is not None else np.array([], dtype=int)
        neighbors.append((row_id, idxs))
        return neighbors


class MedianSplitForestExporter:
    """
    Builds many MedianSplitTree trees (optionally bootstrap-sampled) and optionally exports:
    - nodes.csv: all nodes of all trees, including mean_return
    - tree_features.csv: used characteristics per tree
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        max_features: Any = "sqrt",
        bootstrap: bool = True,
        min_samples_leaf: int = 50,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        max_tries_per_node: int = 25,
        threshold_candidates: Optional[List[float]] = None,
        ):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state
        self.n_jobs = int(n_jobs)
        self.max_tries_per_node = int(max_tries_per_node)
        self.threshold_candidates = threshold_candidates
        self._rng = np.random.RandomState(random_state)
        self.trees_: List[MedianSplitTree] = []

    def fit_and_export(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
        out_nodes_csv: Optional[str] = None,
        out_return_csv: Optional[str] = None
    ) -> None:
       
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y = np.asarray(y).astype(float)
        n = len(X)

        # month keys for bootstrap sampling; keep full array for indexing, but drop NaT for unique choices
        month_keys_full = month_keys(X["date"])
        uniq_months = np.unique(month_keys_full)
        uniq_months = uniq_months[uniq_months != "NaT"]

        # precompute indices per month to avoid repeated np.where inside the estimator loop
        month_to_idx: Dict[str, np.ndarray] = {}
        for m in uniq_months:
            month_to_idx[m] = np.where(month_keys_full == m)[0]

        seeds = self._rng.randint(0, 2**31 - 1, size=self.n_estimators)

        # Clear trees list before new fit
        self.trees_ = []

        nodes_content = []
        excess_return_cols = []
        sampled_month_rows: List[Dict[str, Any]] = []

        rf_series = get_rf_series()

        for i, seed in enumerate(seeds):
            rng = np.random.RandomState(int(seed))
            if self.bootstrap:
                # Sample months with replacement and collect all associated row indices
                sampled_months = rng.choice(uniq_months, size=uniq_months.size, replace=True)
                idx_list = [month_to_idx[mkey] for mkey in sampled_months if mkey in month_to_idx]
                idx = np.concatenate(idx_list) if idx_list else np.array([], dtype=int)
            else:
                idx = np.arange(n)
                sampled_months = uniq_months

            tree = MedianSplitTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(seed),
                threshold_candidates=self.threshold_candidates,
                max_tries_per_node=self.max_tries_per_node,
                rf_series=rf_series
            )
            print("fitting tree", i)
            tree.fit(X, y, idx, sampled_months, feature_cols)
            self.trees_.append(tree)

            if out_nodes_csv or out_return_csv:
                nodes_df, excess_df = tree.export_tables(tree_id=i)
                nodes_content.append(nodes_df)

                # Each excess_df is appended column-wise; prefix columns with tree_id for unique naming
                if excess_df is not None and not excess_df.empty:
                    prefixed = excess_df.rename(columns={c: f"tree{i}_{c}" for c in excess_df.columns})
                    excess_return_cols.append(prefixed)

        if out_nodes_csv or out_return_csv:
            nodes_all = pd.concat([r for r in nodes_content], ignore_index=True)
            # Column-wise combine all excess_df; align lengths via reindex
            if excess_return_cols:
                max_rows = max(df.shape[0] for df in excess_return_cols)
                aligned = [df.reindex(range(max_rows)) for df in excess_return_cols]
                excess_all = pd.concat(aligned, axis=1)
            else:
                excess_all = pd.DataFrame()
                
            # simple export of all nodes (long format)
            if out_nodes_csv:
                nodes_all.to_csv(out_nodes_csv, index=False)

            # export monthly excess returns (wide format)
            if out_return_csv and not excess_all.empty:
                excess_all.to_csv(out_return_csv, index=False)

    def _ensure_fitted(self) -> None:
        if not self.trees_:
            raise ValueError("No trees available. Fit the forest first.")

    def fit_test(
        self,
        X_test: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """
        Uses the trained trees' stored splits to bucket test rows
        """
        self._ensure_fitted()

        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        for tid, tree in enumerate(self.trees_):
            tree.result(X_test, feature_cols) 

    def top_k_neighbors(
        self,
        k: int,
        X: pd.DataFrame,
        feature_cols: List[str],
        y: pd.Series,
        row_id: int,
        mean_output_df: Optional[pd.DataFrame] = None,
        return_mean_output_df: bool = False,
    ):
        """
        Find top-k neighbors per month and optionally export monthly CSVs.

        Process (per query row):
        - Route the provided ``row_id`` through each tree to obtain leaf neighbors.
        - Filter neighbors by month, weight them per tree, and aggregate weights across trees.
        - Then apply top-k truncation and compute the mean return per month.

        Optional outputs (row-based):
        - mean_output_df: Optional DataFrame accumulator for batch runs. If provided,
          the new column block will be joined into the accumulator.
        - return_mean_output_df: If True, the method additionally returns the
          updated accumulator.
        """

        if not self.trees_:
            raise ValueError("No trees available. Fit the forest first.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        month_key_arr = month_keys(X["date"])

        valid_mask = month_key_arr != "NaT"
        unique_months = np.unique(month_key_arr[valid_mask])
        if unique_months.size == 0:
            raise ValueError("No valid months found in date column.")

        # integer-encode months for fast equality checks; invalid months -> -1
        month_codes = pd.Categorical(month_key_arr, categories=unique_months).codes

        n_trees = len(self.trees_)
        X_feat = X[feature_cols]

        # Collect weights per month (row_id only)
        top_k_mean_returns: Dict[int, float] = {}

        tree_neighbors_raw: List[List[Tuple[int, np.ndarray]]] = []
        for tree in self.trees_:
            tree_neighbors_raw.append(tree.get_leaf_neighbors(X_feat, feature_cols, row_id=row_id))

        # Group weights by month:
        monthly_weights_by_month: Dict[int, Dict[int, float]] = {}
        for neighbor_list in tree_neighbors_raw:
            if not neighbor_list:
                continue
            _, idxs = neighbor_list[0]

            if idxs is None or idxs.size == 0:
                print(f"Warning: No neighbors for row_id {row_id} in this tree, skipping.")
                continue

            codes = month_codes[idxs]
            valid_codes_mask = codes >= 0
            if not np.any(valid_codes_mask):
                print(f"Warning: No valid months for row_id {row_id} in this tree, skipping.")
                continue

            idxs_valid = idxs[valid_codes_mask]
            codes_valid = codes[valid_codes_mask]

            for m_code in np.unique(codes_valid):
                idxs_same_month = idxs_valid[codes_valid == m_code]
                if idxs_same_month.size == 0:
                    month_val = unique_months[m_code] if 0 <= m_code < unique_months.size else "NaT"
                    print(f"Warning: No neighbors in the same month ({month_val}) for row_id {row_id} in this tree, skipping.")
                    continue

                uniq_idx, counts = np.unique(idxs_same_month, return_counts=True)
                ho_sum = counts.sum()
                if ho_sum == 0:
                    month_val = unique_months[m_code] if 0 <= m_code < unique_months.size else "NaT"
                    print(f"Warning: Sum of weights for row_id {row_id} in this tree is 0 (month {month_val}), skipping.")
                    continue

                weights = (counts.astype(float) / ho_sum) / n_trees
                w_dict = monthly_weights_by_month.setdefault(int(m_code), {})

                for idx_val, w_add in zip(uniq_idx, weights):
                    w_dict[int(idx_val)] = w_dict.get(int(idx_val), 0.0) + float(w_add)

        # Top-k truncation & mean returns per query row for all months
        top_items_by_month: Dict[str, List[Tuple[int, float]]] = {}
        y_arr_full = np.asarray(y).astype(float)
        for m_code, w_dict in monthly_weights_by_month.items():
            if not w_dict:
                continue

            items = list(w_dict.items())  # (idx, weight)

            if k is None or k <= 0 or len(items) <= k:
                top_items = items
                print(f"Info: For row_id {row_id} and month code {m_code} there are only {len(items)} neighbors, less than k={k}, so no truncation.")
            else:
                # find the top k items by weight without fully sorting
                weights_arr = np.array([wt for _, wt in items])
                idx_arr = np.array([idx for idx, _ in items])
                top_sel = np.argpartition(weights_arr, -k)[-k:]
                top_items = [(int(idx_arr[i]), float(weights_arr[i])) for i in top_sel]

            # normalize weights of the top-k neighbors
            total = sum(w for _, w in top_items)
            if total <= 0:
                continue
            norm_items = [(idx, w / total) for idx, w in top_items]

            mean_ret = float(sum(w * y_arr_full[idx] for idx, w in norm_items))
            top_k_mean_returns[m_code] = mean_ret

            month_val = unique_months[m_code] if 0 <= m_code < unique_months.size else "NaT"
            top_items_by_month[str(month_val)] = [
                (int(idx), float(wt))
                for idx, wt in sorted(norm_items, key=lambda x: x[1], reverse=True)
            ]

        combined_df = mean_output_df if mean_output_df is not None else pd.DataFrame()

        if top_k_mean_returns:
            csv_rows = []
            for m_code, mean_ret in top_k_mean_returns.items():
                month_val = unique_months[m_code] if 0 <= m_code < unique_months.size else "NaT"
                if month_val == "NaT" or pd.isna(month_val):
                    print(f"Warning: Invalid month for row_id {row_id}, skipping.")
                    continue
                csv_rows.append({"month": month_val, "row_id": row_id, "mean": mean_ret})

            if csv_rows:
                wide_df = pd.DataFrame(csv_rows)
                wide_df = wide_df.pivot_table(index="month", columns="row_id", values="mean", aggfunc="first")
                wide_df = wide_df.sort_index()
                wide_df.index = wide_df.index.astype(str)
                wide_df.columns = wide_df.columns.map(str)

                base_df = mean_output_df.copy() if mean_output_df is not None else combined_df
                base_df.index = base_df.index.astype(str)
                base_df.columns = base_df.columns.map(str)
                combined_df = base_df.join(wide_df, how="outer").sort_index()

        if return_mean_output_df:
            return top_items_by_month, combined_df

        return top_items_by_month









