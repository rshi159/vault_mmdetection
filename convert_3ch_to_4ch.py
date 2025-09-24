#!/usr/bin/env python3
import os, sys, argparse, torch
from pathlib import Path

def find_first_3ch_conv(sd, prefix='backbone'):
    """
    Find the earliest backbone conv weight with shape [out, 3, k, k], k>1.
    Returns the key string, or None if not found.
    """
    candidates = []
    for k, w in sd.items():
        if not isinstance(w, torch.Tensor):
            continue
        if k.startswith(prefix) and w.ndim == 4 and w.shape[1] == 3:
            k_h, k_w = w.shape[2], w.shape[3]
            if k_h > 1 and k_w > 1:  # skip 1x1 convs
                candidates.append(k)
    # Choose the lexicographically earliest path (often the stem), but warn if >1.
    if len(candidates) > 1:
        print(f"⚠️  Found multiple 3→4 candidates, picking first:\n    " +
              "\n    ".join(candidates))
    return candidates[0] if candidates else None

def inflate_conv_weight(w):
    # w: [out_c, 3, k, k] -> [out_c, 4, k, k] with zeroed 4th slice
    zero = torch.zeros(w.shape[0], 1, w.shape[2], w.shape[3], dtype=w.dtype)
    return torch.cat([w, zero], dim=1)

def maybe_inflate_block(ckpt, key_name='state_dict'):
    if key_name not in ckpt:
        return False, None, None
    sd = ckpt[key_name]
    conv_key = find_first_3ch_conv(sd, prefix='backbone')
    if conv_key is None:
        return False, None, None
    old = sd[conv_key]
    if old.shape[1] != 3:
        return False, conv_key, old.shape
    new = inflate_conv_weight(old)
    sd[conv_key] = new
    return True, conv_key, (old.shape, new.shape)

def main():
    ap = argparse.ArgumentParser(description="Inflate 3-channel RTMDet/CSPNeXt ckpt to 4-channel (add zero heatmap channel).")
    ap.add_argument("--in", dest="inp", required=True, help="Input .pth checkpoint (3ch)")
    ap.add_argument("--out", dest="out", required=True, help="Output .pth checkpoint (4ch)")
    args = ap.parse_args()

    inp, outp = args.inp, args.out
    print(f"🔧 Converting 3→4 channels\n   • Input:  {inp}\n   • Output: {outp}")
    if not os.path.exists(inp):
        print(f"❌ Input not found: {inp}")
        sys.exit(1)

    ckpt = torch.load(inp, map_location="cpu")
    touched = False
    results = []

    # Handle common blocks
    for block in ["state_dict", "ema_state_dict", "teacher_state_dict"]:
        ok, key, shapes = maybe_inflate_block(ckpt, block)
        if ok:
            touched = True
            results.append((block, key, shapes))
        elif key is not None and isinstance(shapes, tuple):
            # Found but shape not 3ch – probably already 4ch
            print(f"ℹ️  {block}: '{key}' exists but is not 3ch (shape={shapes}); skipping.")

    # Fallback: some checkpoints store weights at top level
    if not touched and all(k not in ckpt for k in ["state_dict", "ema_state_dict", "teacher_state_dict"]):
        ok, key, shapes = maybe_inflate_block({"state_dict": ckpt}, "state_dict")
        if ok:
            ckpt = {"state_dict": ckpt}  # normalize format
            touched = True
            results.append(("state_dict", key, shapes))

    if not touched:
        print("❌ Could not find a 3-input conv under 'backbone' to inflate.")
        sys.exit(1)

    for block, key, (old_shape, new_shape) in results:
        print(f"✅ {block}: inflated {key}  {old_shape} → {new_shape}")

    # Tag metadata
    meta = ckpt.get("meta", {})
    meta["inflated_from_3ch"] = True
    ckpt["meta"] = meta

    Path(os.path.dirname(outp) or ".").mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, outp)
    print("💾 Saved:", outp)
    print("🎉 Done. Ready to train 4-ch with zero heatmap.")

if __name__ == "__main__":
    main()