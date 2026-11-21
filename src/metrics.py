import numpy as np
from scipy.stats import gmean
from gudhi.wasserstein import wasserstein_distance


def match_cycle_across_spaces(orig_cycle, orig_skeleton, sub_pipeline, sub_cycles):
    """Used for ablation analysis, mathes cycles when removing dimenions"""
    orig_verts = set(orig_cycle["vertices"])
    orig_skel = set(orig_skeleton)
    orig_pers = orig_cycle["persistence"]
    orig_b, orig_d = orig_cycle["birth"], orig_cycle["death"]

    if not sub_cycles or len(orig_verts) == 0:
        return {
            "matched_cycle_id": None,
            "full_overlap": 0.0,
            "skeleton_overlap": 0.0,
            "length_diff_pct": 100.0,
            "interval_overlap_pct": 0.0,
            "orig_interval": f"[{orig_b:.4f}, {orig_d:.4f}]",
            "sub_interval": "None",
        }

    best_match = None
    min_cost = float("inf")

    for sc in sub_cycles:
        sc_verts = set(sc["vertices"])
        intersection = orig_verts.intersection(sc_verts)
        subset_overlap = len(intersection) / max(1, len(orig_verts))

        sc_pers = sc["persistence"]
        pers_diff = abs(sc_pers - orig_pers) / max(1e-6, orig_pers)
        birth_diff = abs(sc["birth"] - orig_b) / max(1e-6, orig_b)

        sc_b, sc_d = sc["birth"], sc["death"]
        overlap_start = max(orig_b, sc_b)
        overlap_end = min(orig_d, sc_d)
        overlap_len = max(0.0, overlap_end - overlap_start)

        union_approx = (orig_d - orig_b) + (sc_d - sc_b) - overlap_len
        interval_overlap_pct = (
            round((overlap_len / union_approx) * 100, 1) if union_approx > 0 else 0.0
        )

        cost = pers_diff + (0.5 * birth_diff) - (1.0 * subset_overlap)

        if cost < min_cost:
            min_cost = cost
            sub_skel = set(
                sub_pipeline.extract_minimal_skeleton(sc, protect_critical=True)
            )
            skel_intersection = orig_skel.intersection(sub_skel)
            skel_overlap = len(skel_intersection) / max(1, len(orig_skel))

            best_match = {
                "matched_cycle_id": sc["id"],
                "full_overlap": round(subset_overlap * 100, 1),
                "skeleton_overlap": round(skel_overlap * 100, 1),
                "length_diff_pct": round(pers_diff * 100, 1),
                "interval_overlap_pct": interval_overlap_pct,
                "orig_interval": f"[{orig_b:.4f}, {orig_d:.4f}]",
                "sub_interval": f"[{sc_b:.4f}, {sc_d:.4f}]",
            }

    return best_match


def calculate_cycle_weight(b, p, max_p, filter_mode, custom_expr):
    """Used to give explicit cycles responsible for persistence image coloring."""
    if filter_mode == "None (Constant = 1.0)":
        return 1.0
    elif filter_mode == "Persistence Length (d - b)":
        return p
    elif filter_mode == "Normalized Persistence":
        return p / max_p if max_p > 0 else p
    elif filter_mode == "Custom Formula":
        try:
            return float(
                eval(custom_expr, {"__builtins__": {}}, {"b": b, "p": p})
            )
        except Exception:
            return p
    return p


def generate_persistence_image(
    cycles,
    res=25,
    sigma=0.008,
    filter_mode="Persistence Length (d - b)",
    custom_expr="p",
    fixed_bounds=None,
):
    if not cycles:
        return (
            np.zeros((res, res)),
            np.linspace(0, 1, res),
            np.linspace(0, 1, res),
            {},
        )

    births = [c["birth"] for c in cycles]
    persistences = [c["persistence"] for c in cycles]
    max_p = max(persistences) if persistences else 1.0

    if fixed_bounds is not None:
        b_min, b_max, p_min, p_max = fixed_bounds
    else:
        b_min, b_max = min(births), max(births)
        p_min, p_max = min(persistences), max(persistences)
        b_min, b_max = max(0.0, b_min - 0.005), b_max + 0.005
        p_min, p_max = max(0.0, p_min - 0.005), p_max + 0.005

    b_grid = np.linspace(b_min, b_max, res)
    p_grid = np.linspace(p_min, p_max, res)

    image = np.zeros((res, res))
    cell_mapping = {(r, c): [] for r in range(res) for c in range(res)}

    for c in cycles:
        b_i = c["birth"]
        p_i = c["persistence"]
        weight = calculate_cycle_weight(b_i, p_i, max_p, filter_mode, custom_expr)

        for r in range(res):
            for col in range(res):
                mu_b = b_grid[col]
                mu_p = p_grid[res - 1 - r]

                dist_sq = (b_i - mu_b) ** 2 + (p_i - mu_p) ** 2
                gaussian_val = np.exp(-dist_sq / (2.0 * (sigma**2)))
                contrib = weight * gaussian_val
                image[r, col] += contrib

                if dist_sq <= (2.0 * sigma) ** 2:
                    cell_mapping[(r, col)].append(
                        {
                            "id": c["id"],
                            "birth": round(b_i, 5),
                            "persistence": round(p_i, 5),
                            "contribution": round(contrib, 5),
                        }
                    )

    return image, b_grid, p_grid, cell_mapping


def compute_landscape_layers(cycles, t_grid, n_layers=1):
    layers = np.zeros((n_layers, len(t_grid)))
    if not cycles or len(t_grid) == 0:
        return layers

    for t_idx, t in enumerate(t_grid):
        heights = []
        for c in cycles:
            b, d = c["birth"], c["death"]
            peak = (b + d) / 2.0
            if b <= t <= peak:
                heights.append(t - b)
            elif peak < t <= d:
                heights.append(d - t)
        if heights:
            heights.sort(reverse=True)
            for k in range(min(n_layers, len(heights))):
                layers[k, t_idx] = heights[k]
    return layers


def compute_landscape_lp_norm(cycles, num_pts=200, n_layers=3, t_bounds=None, p=2.0):
    if not cycles:
        return 0.0
    if t_bounds is not None:
        t_min, t_max = t_bounds
    else:
        t_min = min(c["birth"] for c in cycles)
        t_max = max(c["death"] for c in cycles)
    if t_min >= t_max:
        return 0.0

    t_grid = np.linspace(t_min, t_max, num_pts)
    layers = compute_landscape_layers(cycles, t_grid, n_layers=n_layers)
    
    total_p_integral = 0.0
    for k in range(n_layers):
        # Integrate absolute value raised to the power p over the grid
        total_p_integral += float(np.trapezoid(np.abs(layers[k]) ** p, t_grid))
        
    # Raise the total integral to the power of (1 / p)
    return float(total_p_integral ** (1.0 / p))


def compute_persistence_entropy(cycles) -> float:
    """Computes Shannon persistence entropy from the persistence pairs."""
    if not cycles:
        return 0.0
    persistences = np.array(
        [
            c["persistence"]
            for c in cycles
            if not np.isinf(c["persistence"]) and c["persistence"] > 0
        ]
    )
    if len(persistences) == 0:
        return 0.0
    total_p = np.sum(persistences)
    if total_p == 0:
        return 0.0
    probs = persistences / total_p
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return float(entropy)


def compute_betti_curve(cycles, num_pts=150, t_bounds=None):
    if not cycles:
        if t_bounds is not None and t_bounds[0] < t_bounds[1]:
            t_grid = np.linspace(t_bounds[0], t_bounds[1], num_pts)
            return t_grid, np.zeros(num_pts, dtype=int)
        return np.array([]), np.array([])

    valid_pairs = [(c["birth"], c["death"]) for c in cycles if not np.isinf(c["death"])]
    if not valid_pairs:
        return np.array([]), np.array([])

    births = np.array([p[0] for p in valid_pairs])
    deaths = np.array([p[1] for p in valid_pairs])

    if t_bounds is not None:
        t_min, t_max = t_bounds
    else:
        t_min, t_max = min(births), max(deaths)

    if t_min >= t_max:
        return np.array([t_min]), np.array([len(valid_pairs)])

    t_grid = np.linspace(t_min, t_max, num_pts)
    betti_vals = np.array(
        [int(np.sum((births <= t) & (deaths > t))) for t in t_grid]
    )
    return t_grid, betti_vals


def compute_diagram_wasserstein(d1: np.ndarray, d2: np.ndarray, order: float = 1.0) -> float:
    if len(d1) == 0 and len(d2) == 0:
        return 0.0
    else:
        return float(wasserstein_distance(d1, d2, order=order))


def compute_psd_statistics(f: np.ndarray, pxx: np.ndarray) -> dict:
    """Computes summary spectral statistics from Welch's PSD frequencies and power densities."""
    if len(f) == 0 or len(pxx) == 0 or np.sum(pxx) == 0:
        return {
            "Total Spectral Power": 0.0,
            "Peak Frequency (Hz)": 0.0,
            "Peak Spectral Power": 0.0,
            "Spectral Centroid (Hz)": 0.0,
            "Spectral Entropy (Bits)": 0.0,
            "Spectral Flatness": 0.0,
        }
    total_power = (
        float(np.trapezoid(pxx, f)) if len(f) > 1 else float(np.sum(pxx))
    )
    peak_idx = int(np.argmax(pxx))
    peak_freq = float(f[peak_idx])
    peak_power = float(pxx[peak_idx])

    pxx_sum = np.sum(pxx)
    centroid = float(np.sum(f * pxx) / pxx_sum)

    probs = pxx / pxx_sum
    spec_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    pxx_pos = pxx[pxx > 0]
    if len(pxx_pos) > 0:
        spec_flatness = float(gmean(pxx_pos) / np.mean(pxx_pos))
    else:
        spec_flatness = 0.0

    return {
        "Total Spectral Power": round(total_power, 6),
        "Peak Frequency (Hz)": round(peak_freq, 5),
        "Peak Spectral Power": round(peak_power, 6),
        "Spectral Centroid (Hz)": round(centroid, 5),
        "Spectral Entropy (Bits)": round(spec_entropy, 5),
        "Spectral Flatness": round(spec_flatness, 5),
    }