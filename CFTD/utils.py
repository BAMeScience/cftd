import json
import warnings
from pathlib import Path

from ase.filters import UnitCellFilter
from ase.optimize import FIRE
from mace.calculators import mace_mp
import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.core import  Structure, Element
from pymatgen.core.operations import SymmOp
from pymatgen.io.ase import AseAtomsAdaptor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt




'''
Helper file for pymatgen utils and deformation experiments.
'''


'''
Functions to load the structures and transform them into a graph structure
'''
def read_structure_from_csv(filename: str):
    """
    simple csv reader, given a cif input structure.
    """
    df = pd.read_csv(filename, index_col=0)
    structures = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in df.iterrows():
            structures.append(Structure.from_str(row["cif"], fmt="cif"))
    print(f"Parsed {len(structures)} structures from {filename}.")
    return structures

# Optional for JSON structures, not used by us
def load_structures_from_json_column(df, col="structure"):
    structures = []
    for i, val in enumerate(df[col]):
        try:
            s_dict = json.loads(val)
            s = Structure.from_dict(s_dict)
            structures.append(s)
        except Exception as e:
            print(f"Skipping row {i}: {type(e).__name__} - {e}")
    return structures

def structure_to_graph(structure, cutoff=5.0, num_rbf=128, gamma=20.0):
    """
    Encode the pymatgen structure into a graph for the GNN input.
    Uses pymatgen's fast neighbor list (C-backed), safe for skewed lattices.
    """
    if len(structure) == 0:
        raise ValueError("Cannot build a graph for an empty structure.")

    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)

    centers = torch.linspace(0, cutoff, num_rbf, dtype=torch.float)

    edge_index = []
    edge_attr = []
    edge_weight = []
    edge_shift = []

    # ---- FAST, VECTORIZED pymatgen neighbor search ----
    idx_i, idx_j, offsets, dists = structure.get_neighbor_list(
        r=cutoff,
        sites=structure.sites,
        numerical_tol=1e-8,
    )

    lattice_matrix = torch.tensor(
        structure.lattice.matrix, dtype=torch.float
    )

    for i, j, offset, d in zip(idx_i, idx_j, offsets, dists):
        edge_index.append([int(i), int(j)])
        edge_weight.append(float(d))

        # RBF embedding
        edge_attr.append(torch.exp(-gamma * (d - centers) ** 2))

        # periodic image shift → cartesian
        shift_vec = torch.matmul(
            torch.tensor(offset, dtype=torch.float),
            lattice_matrix
        )
        edge_shift.append(shift_vec)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        edge_shift = torch.stack(edge_shift)
    else:
        num_edges = 0
        edge_index = torch.zeros((2, num_edges), dtype=torch.long)
        edge_attr = torch.zeros((num_edges, num_rbf), dtype=torch.float)
        edge_weight = torch.zeros((num_edges,), dtype=torch.float)
        edge_shift = torch.zeros((num_edges, 3), dtype=torch.float)

    cell = lattice_matrix.unsqueeze(0)
    return Data(
        x=z,
        z=z,
        pos=pos,
        cell=cell,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
        edge_shift=edge_shift,
    )



'''
two helper functions to compare CFTD to the more classical novelty and coverage
our CFTD should capture both
IN FEATURE SPACE
'''
def novelty_score(gen_feats, train_feats, threshold=0.05):
    """
    Fraction of generated samples farther than `threshold` from all training samples.
    """
    D = torch.cdist(gen_feats, train_feats, p=2)
    min_dists = D.min(dim=1).values
    return (min_dists > threshold).float().mean().item()


def coverage_score(train_feats, gen_feats, threshold=0.05):
    """
    Fraction of training samples that have at least one generated sample within `threshold`.
    """
    D = torch.cdist(train_feats, gen_feats, p=2)
    min_dists = D.min(dim=1).values
    return (min_dists <= threshold).float().mean().item()


'''
Function to define the Goldilocks zone based to the EMD instead of pairwise-distances;
Used to define all memorized and low-quality structures
'''
def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> torch.Tensor:
    values = values.flatten()
    weights = weights.flatten()

    mask = weights > 0
    values = values[mask]
    weights = weights[mask]

    order = torch.argsort(values)
    values = values[order]
    weights = weights[order]

    cdf = torch.cumsum(weights, dim=0)
    cdf = cdf / cdf[-1]

    idx = torch.searchsorted(
        cdf,
        torch.tensor(q, dtype=cdf.dtype, device=cdf.device),
    )
    idx = torch.clamp(idx, max=values.numel() - 1)

    return values[idx]

'''
Different augment functions
'''

def augment(structure: Structure) -> Structure:
    """
    return an augmented copy of pymatgen struc, only including random
    rotations (determinant 1) and random translation.
    does not change the underlying material distances.
    """
    s = structure.copy()

    # random rotation
    rand_matrix = np.random.normal(size=(3, 3))
    Q, _ = np.linalg.qr(rand_matrix)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    # rotate both lattice and atoms
    op = SymmOp.from_rotation_and_translation(
        rotation_matrix=Q, translation_vec=[0, 0, 0]
    )
    s.apply_operation(op, fractional=False)  # apply in cartesian space

    # random fractional translation
    shift = np.random.rand(3)
    s.translate_sites(range(len(s)), shift, frac_coords=True, to_unit_cell=True)

    return s

def augment_similar(
    structure: Structure,
    modified_pettifor_scale: dict,
    allowed_elements: set,
    probability_nearest_neighbor: float = 0.1,
    probability_next_nearest_neighbor: float = 0.05,
    p_noise: float = 0.2,
    noise_scale: float = 0.005,
) -> Structure:
    """
    Return rotated,translated version of structure where, with a certain
    probability all sites of one element are replaced with either nearest or
    next-nearest neighbor on the modified Pettifor scale introduced in
    Glawe et al. 10.1088/1367-2630/18/9/093011.

    With probability p_noise, add wrapped Gaussian noise in fractional
    coordinates (std=noise_scale).
    """
    structure_copy = structure.copy()
    structure_copy.remove_oxidation_states()  # important f. matching below

    element_symbols = [el.symbol for el in structure_copy.elements]
    eligible_symbols = [
        el for el in element_symbols if el in modified_pettifor_scale and el in allowed_elements
    ]
    if len(eligible_symbols) == 0:
        # No valid Pettifor substitution candidate in this structure -> keep composition unchanged.
        augmented = augment(structure=structure_copy)
        if np.random.rand() < float(p_noise):
            frac = np.array([site.frac_coords for site in augmented.sites], dtype=float)
            noise = np.random.normal(loc=0.0, scale=float(noise_scale), size=frac.shape)
            frac_noisy = (frac + noise) % 1.0
            augmented = Structure(
                lattice=augmented.lattice,
                species=[site.specie.symbol for site in augmented.sites],
                coords=frac_noisy,
                coords_are_cartesian=False,
            )
        return augmented

    el_to_replace = np.random.choice(a=eligible_symbols, size=None)
    neighbor_els = []

    if np.random.rand() < probability_nearest_neighbor:
        neighbor_els = get_pettifor_neighbors(
            elem=el_to_replace,
            modified_pettifor_scale=modified_pettifor_scale,
            allowed_elems=allowed_elements,
            is_next_nearest=False,
        )

    elif np.random.rand() < probability_next_nearest_neighbor:
        neighbor_els = get_pettifor_neighbors(
            elem=el_to_replace,
            modified_pettifor_scale=modified_pettifor_scale,
            allowed_elems=allowed_elements,
            is_next_nearest=True,
        )

    if neighbor_els:
        replacing_el = np.random.choice(a=neighbor_els, size=None)
        structure_copy[el_to_replace] = replacing_el

    # Rotate and translate (additionally: supercells?)
    augmented = augment(structure=structure_copy)
    if np.random.rand() < float(p_noise):
        frac = np.array([site.frac_coords for site in augmented.sites], dtype=float)
        noise = np.random.normal(loc=0.0, scale=float(noise_scale), size=frac.shape)
        frac_noisy = (frac + noise) % 1.0
        augmented = Structure(
            lattice=augmented.lattice,
            species=[site.specie.symbol for site in augmented.sites],
            coords=frac_noisy,
            coords_are_cartesian=False,
        )
    return augmented


'''
Helpers for augment_similar
''' 
def load_modified_pettifor_scale(json_path: str) -> dict:
    """
    Load the modified Pettifor scale mapping element symbol -> order index.
    """
    with open(json_path, "r") as f:
        return json.load(f)

def get_pettifor_neighbors(
    elem: str,
    modified_pettifor_scale: dict,
    allowed_elems: set,
    is_next_nearest: bool = False,
) -> list:
    """
    For a given element, return its nearest or next-nearest neighbors on the
    modified Pettifor scale.

    If at the ends, neighbors are one-sided.

    If the nearest or next-nearest neighbor is not included in the
    dataset return only other neighbor.

    If both nearest or next-nearest neighbors are not included in the
    dataset return empty list -> no element replacement will be made in augment_similar

    If candidates are out of bounds or not in allowed_elems, they are ignored.
    """
    if elem not in modified_pettifor_scale or elem not in allowed_elems:
        return []

    sorted_elements = sorted(
        modified_pettifor_scale.keys(), key=lambda e: modified_pettifor_scale[e]
    )

    idx = sorted_elements.index(elem)

    neighbors = []

    step = 2 if is_next_nearest else 1
    neighbor_indices = [idx - step, idx + step]

    for n_idx in neighbor_indices:
        if 0 <= n_idx < len(sorted_elements):
            neighbor_element = sorted_elements[n_idx]
            if neighbor_element in allowed_elems:
                neighbors.append(neighbor_element)

    return neighbors

def augment_with_frac_noise(
    s: Structure,
    *,
    noise_std: float = 0.02,
    clip: float | None = None,
) -> Structure:
    """
    Add i.i.d. Gaussian noise to fractional coords and wrap into [0, 1).
    noise_std is in fractional coordinate units.

    clip: if not None, clip each component of the noise to [-clip, clip]
          (also in fractional units), which can reduce extreme displacements.
    """

    frac = np.array([site.frac_coords for site in s.sites], dtype=float)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=frac.shape)

    if clip is not None:
        noise = np.clip(noise, -clip, clip)

    frac_noisy = (frac + noise) % 1.0  # wrap into [0, 1)
    str = Structure(
        lattice=s.lattice,
        species=[site.specie.symbol for site in s.sites],
        coords=frac_noisy,
        coords_are_cartesian=False,
    )
    return augment(str)


'''
Functions used for the deformation experiments
'''
def perturb_structures_gaussian(
    original_structures, sigma=0.05, rng=None
):
    """
    perturbs structures (fractional space) with gaussian noise of standard deviation sigma
    note that sigma = 0.05 is already quite large! (think 0.05 of gaussian in (-3,3))
    proxy for stability
    """

    perturbed_structures = []

    for original in original_structures:
        new_coords = []

        for site in original.sites:
            u = np.asarray(site.frac_coords, dtype=float)

            # Gaussian noise in fractional space, then wrap to [0,1)
            noise = np.random.normal(0.0, sigma, size=3)
            # mod 1 to stay in frac coordinates
            pert = (u + noise) % 1.0

            new_coords.append(pert)
        # set new structure, species stay fixed, lattice to, but frac coords change
        perturbed = Structure(
            lattice=original.lattice,
            species=[s.species for s in original.sites],
            coords=new_coords,
            coords_are_cartesian=False,
        )
        perturbed_structures.append(perturbed)

    return perturbed_structures

def random_lattice_deformation(
    s: Structure,
    max_strain: float = 0.1) -> Structure:
    """
    Apply uniform lattice deformation with intensity max_strain:
    either all axes expand or all axes contract together.
    """

    A = s.lattice.matrix

    # One shared sign for all axes.
    sign = float(np.random.choice([-1, 1]))
    scale = 1.0 + sign * max_strain
    new_lat = A * scale

    return Structure(
        lattice=new_lat,
        species=[str(site.specie) for site in s.sites],
        coords=[site.frac_coords for site in s.sites],
        coords_are_cartesian=False
    )

def random_supercell(
    s: Structure,
    p: float = 0.5) -> Structure:
    """
    randomly create a supercell with probability p. same scaling as augment_supercell
    note that this does not include rotations or translations
    """
    scale_options = [
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (2, 2, 1),
        (1, 2, 2),
        (2, 1, 2),
        (2, 2, 2),
    ]

    new_struct = s.copy()

    if np.random.random() < p:
        scale = scale_options[np.random.randint(len(scale_options))]
        new_struct.make_supercell(scale)

    return new_struct

def random_group_substitution(
    s: Structure,
    allowed_elements: set,
    p: float = 0.05) -> Structure:
    """
    random substituion within the same "group", i.e. column of periodic table
    """
    
    coords = np.array([site.frac_coords for site in s.sites])
    old_species = np.array([str(site.specie) for site in s.sites], dtype=object)
    new_species = old_species.copy()

    mask = np.random.random(len(old_species)) < p

    for idx in np.where(mask)[0]:
        elem = Element(old_species[idx])
        group = elem.group
        candidates = [
            e for e in allowed_elements
            if Element(e).group == elem.group and e != elem.symbol
        ]

        if candidates:
            new_species[idx] = np.random.choice(candidates)

    return Structure(
        lattice=s.lattice,
        species=new_species.tolist(),
        coords=coords,
        coords_are_cartesian=False,
    )

def random_substitution(
    s: Structure,
    allowed_elements: set,
    p: float = 0.05) -> Structure:
    """
    randomly substitute within any "allowed" elements (contained in some material in train dataset)
    """

    coords = np.array([site.frac_coords for site in s.sites])
    old_species = np.array([str(site.specie) for site in s.sites], dtype=object)
    new_species = old_species.copy()

    mask = np.random.random(len(old_species)) < p

    for idx in np.where(mask)[0]:
        elem = Element(old_species[idx])

        candidates = [
            e for e in allowed_elements
            if e != elem.symbol
        ]

        if candidates:
            new_species[idx] = np.random.choice(candidates)

    return Structure(
        lattice=s.lattice,
        species=new_species.tolist(),
        coords=coords,
        coords_are_cartesian=False,
    )

def random_pettifor_nn_site_substitutions(
    structures: list[Structure],
    modified_pettifor_scale: dict,
    allowed_elements: set,
    p_sub: float,
) -> tuple[list[Structure], float]:
    """
    Replace each eligible site with probability p_sub by a Pettifor nearest neighbor.

    Returns the augmented structures and the fraction of all sites that changed.
    This matches the per-site semantics of random_substitution and
    random_group_substitution.
    """
    out = []
    changed_sites = 0
    total_sites = 0

    for structure in structures:
        species = []
        frac_coords = []
        for site in structure.sites:
            total_sites += 1
            elem = str(site.specie.symbol)
            replacement = elem
            if elem in modified_pettifor_scale and elem in allowed_elements:
                if float(np.random.rand()) < float(p_sub):
                    neighbors = get_pettifor_neighbors(
                        elem=elem,
                        modified_pettifor_scale=modified_pettifor_scale,
                        allowed_elems=allowed_elements,
                        is_next_nearest=False,
                    )
                    if neighbors:
                        replacement = str(np.random.choice(neighbors))
                        changed_sites += int(replacement != elem)
            species.append(replacement)
            frac_coords.append(site.frac_coords)

        out.append(
            Structure(
                lattice=structure.lattice,
                species=species,
                coords=frac_coords,
                coords_are_cartesian=False,
            )
        )

    changed_frac = float(changed_sites) / max(int(total_sites), 1)
    return out, changed_frac


'''
Function for relaxation
'''
def relax_structures(
    structures,
    *,
    mace_model: str = "small",
    device: str = "cuda",
    steps: int = 50,
    fmax: float = 0.03,
):
    adaptor = AseAtomsAdaptor()
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    
    calc = mace_mp(model=str(mace_model), 
                   device=str(device), 
                   default_dtype="float64", 
                   compile=False)
    
    supported_atomic_numbers = {int(z) for z in calc.models[0].atomic_numbers}#
    relaxed = []
    for s in structures:
        structure_atomic_numbers = {int(site.specie.Z) for site in s.sites}
        if not structure_atomic_numbers.issubset(supported_atomic_numbers):
            relaxed.append(s)
            continue
        
        atoms = adaptor.get_atoms(s)
        atoms.pbc = True
        atoms.calc = calc
        
        opt = FIRE(UnitCellFilter(atoms, mask=[1, 1, 1, 1, 1, 1]), logfile=None)
        opt.run(fmax=float(fmax), steps=int(steps))
        
        relaxed.append(adaptor.get_structure(atoms))
        
    torch.set_default_dtype(old_default_dtype)
    return relaxed


'''
Functions to plot different generative models for comparisons
'''
def plot_model_eval_split(rows, out_path, title):
    labels = [str(r["model"]) for r in rows]
    qual = np.array([float(r["cftd_quality"]) for r in rows], dtype=float)
    mem = np.array([float(r["cftd_memorization"]) for r in rows], dtype=float)
    x = np.arange(len(labels), dtype=float)
    
    fig, ax = plt.subplots(figsize=(max(8.0, 0.8 * len(labels)), 5.0))
    ax.bar(x, qual, color="#4C78A8", label="quality term")
    ax.bar(x, mem, bottom=qual, color="#F58518", label="memorization term")
    ax.set_title(title, fontsize=16, wrap=True)
    ax.set_ylabel("CFTD", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    
    print(f"Saved plot: {out_path}")

def savefig_with_zoom(
    path,
    *,
    zoom_n: int = 3,
    dpi: int = 300,
    figsize=(11, 7),
    labelsize: int = 22,
    ticksize: int = 20,
    titlesize: int = 24,
    legendsize: int = 20,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    axes = fig.axes
    fig.set_size_inches(*figsize, forward=True)
    for ax in axes:
        ax.title.set_fontsize(titlesize)
        ax.xaxis.label.set_fontsize(labelsize)
        ax.yaxis.label.set_fontsize(labelsize)
        ax.tick_params(axis="both", labelsize=ticksize)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(legendsize)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)

    if not axes:
        return
    src_ax = axes[0]
    zoom_path = path.with_name(f"{path.stem}_zoom{zoom_n}{path.suffix}")
    zoom_fig, zoom_ax = plt.subplots(figsize=figsize)
    for line in src_ax.get_lines():
        label = line.get_label()
        kwargs = {
            "marker": line.get_marker(),
            "linestyle": line.get_linestyle(),
            "linewidth": line.get_linewidth(),
            "markersize": line.get_markersize(),
            "color": line.get_color(),
        }
        if label and not label.startswith("_"):
            kwargs["label"] = label
        zoom_ax.plot(
            np.asarray(line.get_xdata(orig=False))[:zoom_n],
            np.asarray(line.get_ydata(orig=False))[:zoom_n],
            **kwargs,
        )
    zoom_ax.set_xlabel(src_ax.get_xlabel(), fontsize=labelsize)
    zoom_ax.set_ylabel(src_ax.get_ylabel(), fontsize=labelsize)
    title = src_ax.get_title()
    zoom_ax.set_title(f"{title} (first {zoom_n})" if title else f"First {zoom_n}", fontsize=titlesize)
    zoom_ax.tick_params(axis="both", labelsize=ticksize)
    if src_ax.get_legend() is not None:
        zoom_ax.legend(fontsize=legendsize)
    zoom_ax.grid(True)
    zoom_fig.tight_layout()
    zoom_fig.savefig(zoom_path, dpi=dpi)
    plt.close(zoom_fig)
    plt.figure(fig.number)