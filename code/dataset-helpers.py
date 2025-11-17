IMG_H = 28
IMG_W = 28
FLAT_SIZE = IMG_H * IMG_W  # 784

def _to_grid(x):
    # x: (..., 784) -> (..., 28, 28) (view)
    return x.view(*x.shape[:-1], IMG_H, IMG_W)

def _to_flat(x):
    # x: (..., 28, 28) -> (..., 784) (view/reshape)
    return x.reshape(*x.shape[:-2], FLAT_SIZE)

# ---------- Geometric image transforms (flattened input) ----------
def hflip(x):
    """
    Horizontal flip (left-right).
    """
    g = _to_grid(x)
    g = torch.flip(g, dims=(-1,))          # flip width
    return _to_flat(g)

def vflip(x):
    """
    Vertical flip (top-bottom).
    """
    g = _to_grid(x)
    g = torch.flip(g, dims=(-2,))          # flip height
    return _to_flat(g)

def rot90(x, k=1):
    """
    Rotate 90° clockwise k times (k in {0,1,2,3}).
    Implemented as transpose + flip (no negative-step slicing).
    """
    k %= 4
    if k == 0:
        return x
    g = _to_grid(x)
    for _ in range(k):
        g = torch.flip(g.transpose(-2, -1), dims=(-2,))  # CW: transpose then flip rows
    return _to_flat(g)

# ---------- Count label remaps ----------
# counts layout: [squares, circles, up, right, down, left]
IDX_HFLIP = torch.tensor([0, 1, 2, 5, 4, 3])  # right<->left
IDX_VFLIP = torch.tensor([0, 1, 4, 3, 2, 5])  # up<->down

def remap_hflip(counts):
    return counts[..., IDX_HFLIP]

def remap_vflip(counts):
    return counts[..., IDX_VFLIP]

def remap_rot90(counts, k=1):
    """
    Rotate orientation counts clockwise k times: up->right->down->left.
    """
    head = counts[..., :2]
    orient = counts[..., 2:6]
    orient = torch.roll(orient, shifts=k, dims=-1)  # CW shift
    return torch.cat([head, orient], dim=-1)

# ---------- Combined convenience ----------
def apply_geom(x_flat, counts6,
               do_h=False, do_v=False, k_rot=0):
    """
    Apply geometric transforms to flattened image and counts.
    Order: rotation -> hflip -> vflip (consistent remap).
    Returns (x_aug, counts_aug).
    """
    k_rot %= 4
    if k_rot:
        x_flat = rot90(x_flat, k_rot)
        counts6 = remap_rot90(counts6, k_rot)
    if do_h:
        x_flat = hflip(x_flat)
        counts6 = remap_hflip(counts6)
    if do_v:
        x_flat = vflip(x_flat)
        counts6 = remap_vflip(counts6)
    return x_flat, counts6

def encode_counts_to_class135(counts: torch.Tensor) -> torch.Tensor:
    # counts shape (6,), exactly two non-zero entries
    nz = torch.nonzero(counts > 0, as_tuple=False).flatten()
    a, b = nz.sort().values.tolist()
    row = a * (11 - a) // 2 + (b - a - 1)   # unordered pair index
    col = int(counts[a].item()) - 1         # 0..8
    return torch.tensor(row * 9 + col, dtype=torch.long)