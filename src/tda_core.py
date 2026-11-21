import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
import streamlit as st
import networkx as nx


def extract_window_sub_values(df_values, start_idx, window_size, pc_stride, window_direction, stride_agg_mode):
    n_total = df_values.shape[0]
    if window_direction == "Past (Backward from Start Index)":
        end_idx = min(n_total, start_idx + window_size)
        raw_window_vals = df_values[start_idx:end_idx]
    else:
        start_sub = max(0, start_idx - window_size + 1)
        end_sub = start_idx + 1
        raw_window_vals = df_values[start_sub:end_sub]

    if len(raw_window_vals) == 0:
        return np.empty((0, df_values.shape[1]))

    if stride_agg_mode == "Raw Value (Every n-th row)":
        return raw_window_vals[::pc_stride]
    else:
        indices = list(range(0, len(raw_window_vals), pc_stride))
        aggregated_rows = []
        for idx in indices:
            if "Past" in stride_agg_mode:
                sub_start = max(0, idx - pc_stride + 1)
                chunk = raw_window_vals[sub_start : idx + 1]
            else:
                sub_end = min(len(raw_window_vals), idx + pc_stride)
                chunk = raw_window_vals[idx:sub_end]

            if "Sum" in stride_agg_mode:
                aggregated_rows.append(np.sum(chunk, axis=0))
            else:
                aggregated_rows.append(np.mean(chunk, axis=0))

        return np.array(aggregated_rows)


@st.cache_data
def compute_fast_diagram_only(
    df_values,
    col_indices_tuple,
    selected_dims_tuple,
    start_idx,
    window_size,
    pc_stride,
    window_direction,
    stride_agg_mode,
    standardize,
    homology_dim,
    thresh_type_val,
    thresh_val,
):
    """faster ripser, computes only diagrams."""
    n_total = df_values.shape[0]
    sub_vals = extract_window_sub_values(
        df_values, start_idx, window_size, pc_stride, window_direction, stride_agg_mode
    )

    raw_pc = sub_vals[:, col_indices_tuple].astype(np.float64)

    if standardize and n_total > 1:
        global_mean = np.mean(df_values[:, col_indices_tuple], axis=0)
        global_std = np.std(df_values[:, col_indices_tuple], axis=0)
        global_std[global_std == 0.0] = 1.0
        point_cloud = (raw_pc - global_mean) / global_std
    else:
        point_cloud = raw_pc

    n_points = point_cloud.shape[0]
    min_pts_req = 1 if homology_dim == 0 else (3 if homology_dim == 1 else 6)
    min_dims_req = 1 if homology_dim == 0 else (2 if homology_dim == 1 else 3)

    if n_points < min_pts_req or point_cloud.shape[1] < min_dims_req:
        return np.empty((0, 2)), 0.0, 1.0

    dist_matrix = squareform(pdist(point_cloud))
    triu_vals = dist_matrix[np.triu_indices(n_points, k=1)]
    min_d = float(np.min(triu_vals)) if len(triu_vals) > 0 else 0.0
    max_d = float(np.max(triu_vals)) if len(triu_vals) > 0 else 1.0

    res = ripser(dist_matrix, maxdim=homology_dim, distance_matrix=True, do_cocycles=False)

    if homology_dim >= len(res["dgms"]):
        return np.empty((0, 2)), min_d, max_d

    diagram = res["dgms"][homology_dim]
    valid_pairs = []

    for b, d in diagram:
        if np.isinf(d):
            continue
        persistence = float(d - b)
        valid_pairs.append((float(b), float(d), persistence))

    if not valid_pairs:
        return np.empty((0, 2)), min_d, max_d

    max_p_w = max(p for _, _, p in valid_pairs)
    eff_thresh_w = (
        max_p_w * (thresh_val / 100.0)
        if thresh_type_val == "Relative (% of Max Persistence)"
        else thresh_val
    )

    filtered_diagram = np.array([[b, d] for b, d, p in valid_pairs if p >= eff_thresh_w])
    return filtered_diagram, min_d, max_d


class TDAPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        selected_dims: list,
        start_idx: int,
        window_size: int,
        pc_stride: int = 1,
        stride_agg_mode: str = "Raw Value (Every n-th row)",
        window_direction: str = "Future (Forward from Start Index)",
        standardize: bool = True,
        homology_dim: int = 1,
    ):
        self.df = df
        self.selected_dims = selected_dims
        self.start_idx = start_idx
        self.window_size = window_size
        self.pc_stride = pc_stride
        self.stride_agg_mode = stride_agg_mode
        self.window_direction = window_direction
        self.standardize = standardize
        self.homology_dim = homology_dim

        n_total = len(df)
        df_vals = df.values
        sub_vals = extract_window_sub_values(
            df_vals, start_idx, window_size, pc_stride, window_direction, stride_agg_mode
        )

        if window_direction == "Future (Forward from Start Index)":
            end_idx = min(n_total, start_idx + window_size)
            raw_w_df = df.iloc[start_idx:end_idx]
        else:
            start_sub = max(0, start_idx - window_size + 1)
            end_sub = start_idx + 1
            raw_w_df = df.iloc[start_sub:end_sub]

        if stride_agg_mode == "Raw Value (Every n-th row)":
            self.window_df = raw_w_df.iloc[::pc_stride]
        else:
            indices = list(range(0, len(raw_w_df), pc_stride))
            self.window_df = raw_w_df.iloc[indices]

        raw_pc = sub_vals[:, [df.columns.get_loc(d) for d in selected_dims]].astype(np.float64)

        if self.standardize and n_total > 1:
            full_data = df[selected_dims].values.astype(np.float64)
            global_mean = np.mean(full_data, axis=0)
            global_std = np.std(full_data, axis=0)
            global_std[global_std == 0.0] = 1.0
            self.point_cloud = (raw_pc - global_mean) / global_std
        else:
            self.point_cloud = raw_pc

        self.n_points, self.n_dims = self.point_cloud.shape

        min_pts_req = 1 if homology_dim == 0 else (3 if homology_dim == 1 else 6)
        min_dims_req = 1 if homology_dim == 0 else (2 if homology_dim == 1 else 3)

        if self.n_points >= min_pts_req and self.n_dims >= min_dims_req:
            self.distance_matrix = squareform(pdist(self.point_cloud))
        else:
            self.distance_matrix = None

    def compute_raw_cycles(self):
        min_pts_req = 1 if self.homology_dim == 0 else (3 if self.homology_dim == 1 else 6)
        min_dims_req = 1 if self.homology_dim == 0 else (2 if self.homology_dim == 1 else 3)

        if self.distance_matrix is None or self.n_dims < min_dims_req or self.n_points < min_pts_req:
            return []

        do_cocycles_flag = self.homology_dim >= 1
        res = ripser(
            self.distance_matrix,
            maxdim=self.homology_dim,
            distance_matrix=True,
            do_cocycles=do_cocycles_flag,
        )

        if self.homology_dim >= len(res["dgms"]):
            return []

        diagram = res["dgms"][self.homology_dim]
        cocycles = (
            res["cocycles"][self.homology_dim]
            if (do_cocycles_flag and "cocycles" in res and len(res["cocycles"]) > self.homology_dim)
            else None
        )

        triu_i, triu_j = np.triu_indices(self.n_points, k=1)
        triu_dists = self.distance_matrix[triu_i, triu_j]
        used_h0_vertices = set()

        cycles_info = []
        valid_id_counter = 0

        for idx, (b, d) in enumerate(diagram):
            persistence = float(d - b)
            if np.isinf(d):
                continue

            # -----------------------------------------------------------------
            # HOMOLOGY DIMENSION 0 (Connected Components)
            # -----------------------------------------------------------------
            if self.homology_dim == 0:
                if len(triu_dists) > 0:
                    closest_edge_idx = int(np.argmin(np.abs(triu_dists - d)))
                    u, v = int(triu_i[closest_edge_idx]), int(triu_j[closest_edge_idx])
                    death_edge = [u, v]
                    pre_death_edge = [u, v]
                    birth_pt = v if (u in used_h0_vertices and v not in used_h0_vertices) else u
                    used_h0_vertices.add(birth_pt)
                else:
                    birth_pt = 0
                    death_edge = [0, 0]
                    pre_death_edge = [0, 0]

                full_loop_indices = [birth_pt]
                raw_cocycle_indices = [birth_pt]
                birth_edge = [birth_pt]
                pre_death_neighbors = [pt_idx for pt_idx in range(self.n_points) if self.distance_matrix[birth_pt, pt_idx] <= d * 1.01]
                if not pre_death_neighbors:
                    pre_death_neighbors = [birth_pt]

            # -----------------------------------------------------------------
            # HOMOLOGY DIMENSION 1 
            # -----------------------------------------------------------------
            elif self.homology_dim == 1:
                pre_death_neighbors = []
                raw_cocycle_indices = []
                
                if cocycles is not None and idx < len(cocycles) and len(cocycles[idx]) > 0:
                    edges = cocycles[idx][:, :2].astype(int)
                    raw_cocycle_indices = sorted(list(set(edges.flatten())))

                    # Use NetworkX to find the connected component loop at birth threshold b
                    G_birth = nx.Graph()
                    adj_i, adj_j = np.where((self.distance_matrix <= b * 1.12) & (self.distance_matrix > 0))
                    G_birth.add_edges_from(zip(adj_i.tolist(), adj_j.tolist()))

                    full_loop_indices = raw_cocycle_indices
                    for comp in nx.connected_components(G_birth):
                        if any(v in comp for v in raw_cocycle_indices):
                            full_loop_indices = sorted(list(comp))
                            break
                    if len(full_loop_indices) < 3:
                        full_loop_indices = raw_cocycle_indices
                else:
                    full_loop_indices = [0, 1, 2] if self.n_points >= 3 else list(range(self.n_points))
                    raw_cocycle_indices = full_loop_indices

               # Uniform distance-based edge extraction for the feature vertices
                feat_edges = []
                for i in range(len(full_loop_indices)):
                    for j in range(i + 1, len(full_loop_indices)):
                        u, v = full_loop_indices[i], full_loop_indices[j]
                        feat_edges.append((self.distance_matrix[u, v], int(u), int(v)))
                feat_edges.sort(key=lambda x: x[0])

                # 1. Birth Edge (closest to b)
                birth_edge = [feat_edges[0][1], feat_edges[0][2]] if feat_edges else [full_loop_indices[0], full_loop_indices[1]]
                min_b_diff = float("inf")
                for dist, u, v in feat_edges:
                    if abs(dist - b) < min_b_diff:
                        min_b_diff = abs(dist - b)
                        birth_edge = [u, v]

                # 2. Actual Death Edge (closest to d - the edge that kills the cycle)
                death_edge = birth_edge
                min_d_diff = float("inf")
                for dist, u, v in feat_edges:
                    if abs(dist - d) < min_d_diff:
                        min_d_diff = abs(dist - d)
                        death_edge = [u, v]

                # 3. Pre-Death Edge (The last edge that appeared *before* death d)
                # This is the largest edge in the loop whose distance is strictly less than d
                smaller_edges = [e for e in feat_edges if e[0] < d]
                if smaller_edges:
                    pre_death_edge = [smaller_edges[-1][1], smaller_edges[-1][2]]
                else:
                    pre_death_edge = death_edge

            # -----------------------------------------------------------------
            # HOMOLOGY DIMENSION 2
            # -----------------------------------------------------------------
            else:
                pre_death_neighbors = []
                raw_cocycle_indices = []
                
                if cocycles is not None and idx < len(cocycles) and len(cocycles[idx]) > 0:
                    simplices = cocycles[idx][:, :3].astype(int)
                    raw_cocycle_indices = sorted(list(set(simplices.flatten())))

                G_birth = nx.Graph()
                adj_i, adj_j = np.where((self.distance_matrix <= b * 1.15) & (self.distance_matrix > 0))
                G_birth.add_edges_from(zip(adj_i.tolist(), adj_j.tolist()))

                full_loop_indices = list(raw_cocycle_indices)
                if raw_cocycle_indices:
                    for comp in nx.connected_components(G_birth):
                        if any(v in comp for v in raw_cocycle_indices):
                            full_loop_indices = sorted(list(comp))
                            break

                if len(full_loop_indices) < 6:
                    if len(triu_dists) > 0:
                        closest_b_idx = int(np.argmin(np.abs(triu_dists - b)))
                        seed_u, seed_v = int(triu_i[closest_b_idx]), int(triu_j[closest_b_idx])
                        seed_set = set(full_loop_indices) | {seed_u, seed_v}
                        dists_to_seed = np.min(self.distance_matrix[list(seed_set), :], axis=0)
                        nearest_order = np.argsort(dists_to_seed)
                        for pt in nearest_order:
                            seed_set.add(int(pt))
                            if len(seed_set) >= min(6, self.n_points):
                                break
                        full_loop_indices = sorted(list(seed_set))
                    else:
                        full_loop_indices = list(range(min(6, self.n_points)))

                if not raw_cocycle_indices:
                    raw_cocycle_indices = full_loop_indices

                feat_edges = []
                for i in range(len(full_loop_indices)):
                    for j in range(i + 1, len(full_loop_indices)):
                        u, v = full_loop_indices[i], full_loop_indices[j]
                        feat_edges.append((self.distance_matrix[u, v], int(u), int(v)))
                feat_edges.sort(key=lambda x: x[0])

                # 1. Birth Edge (closest to b)
                birth_edge = [feat_edges[0][1], feat_edges[0][2]] if feat_edges else [full_loop_indices[0], full_loop_indices[1]]
                min_b_diff = float("inf")
                for dist, u, v in feat_edges:
                    if abs(dist - b) < min_b_diff:
                        min_b_diff = abs(dist - b)
                        birth_edge = [u, v]

                # 2. Actual Death Edge (closest to d) and its index
                death_edge = birth_edge
                death_edge_idx = 0
                min_d_diff = float("inf")
                for idx_e, (dist, u, v) in enumerate(feat_edges):
                    if abs(dist - d) < min_d_diff:
                        min_d_diff = abs(dist - d)
                        death_edge = [u, v]
                        death_edge_idx = idx_e

                # 3. Pre-Death Edge (The edge added immediately before the death edge in filtration order)
                if death_edge_idx > 0:
                    pre_death_edge = [feat_edges[death_edge_idx - 1][1], feat_edges[death_edge_idx - 1][2]]
                else:
                    pre_death_edge = death_edge

            # Append structured feature dictionary
            cycles_info.append(
                {
                    "id": valid_id_counter,
                    "birth": float(b),
                    "death": float(d),
                    "persistence": persistence,
                    "peak_time": float((b + d) / 2.0),
                    "peak_height": persistence / 2.0,
                    "vertices": full_loop_indices,
                    "raw_arc": raw_cocycle_indices,
                    "birth_edge": list(birth_edge) if birth_edge else [],
                    "death_edge": list(death_edge) if death_edge else [],
                    "pre_death_edge": list(pre_death_edge) if pre_death_edge else [],
                    "pre_death_neighbors": pre_death_neighbors if self.homology_dim == 0 else [],
                    "n_points": len(full_loop_indices),
                }
            )
            valid_id_counter += 1

        return cycles_info

    def extract_minimal_skeleton(
        self, cycle_info, tol_b=0.20, tol_p=0.25, protect_critical=True
    ):
        if self.homology_dim == 0:
            return cycle_info["vertices"][:1] if cycle_info["vertices"] else []

        min_verts = 3 if self.homology_dim == 1 else 6
        orig_indices = list(cycle_info["vertices"])
        if len(orig_indices) <= min_verts:
            return sorted(orig_indices)

        orig_b = cycle_info["birth"]
        orig_pers = cycle_info["persistence"]

        protected_vertices = set()
        if protect_critical:
            protected_vertices.update(cycle_info.get('birth_edge', []))
            protected_vertices.update(cycle_info.get('death_edge', []))

        current_indices = list(orig_indices)

        for v in orig_indices:
            if len(current_indices) <= min_verts:
                break
            if v in protected_vertices:
                continue

            test_indices = [node for node in current_indices if node != v]
            sub_pc = self.point_cloud[test_indices]

            if len(sub_pc) < min_verts:
                continue

            sub_pipeline = TDAPipeline(
                self.df,
                self.selected_dims,
                self.start_idx,
                self.window_size,
                pc_stride=self.pc_stride,
                stride_agg_mode=self.stride_agg_mode,
                window_direction=self.window_direction,
                standardize=self.standardize,
                homology_dim=self.homology_dim,
            )
            sub_pipeline.point_cloud = sub_pc
            sub_pipeline.n_points = len(sub_pc)
            sub_pipeline.distance_matrix = squareform(pdist(sub_pc))
            sub_cycles = sub_pipeline.compute_raw_cycles()

            preserved = False
            for sc in sub_cycles:
                if (abs(sc["birth"] - orig_b) <= max(0.08, orig_b * tol_b)) and (
                    abs(sc["persistence"] - orig_pers) <= max(0.08, orig_pers * tol_p)
                ):
                    preserved = True
                    break

            if preserved:
                current_indices = test_indices

        return sorted(current_indices)