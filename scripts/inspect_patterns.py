import numpy as np, sys
sys.path.insert(0, 'src')
d = np.load('/tmp/pooled_ss.npz', allow_pickle=True)
X, y = d['X'], d['y']
pmap = d['pattern_map'].item() if 'pattern_map' in d.files else {}
# Reverse map: remapped_idx -> original pattern int
rev = {v: int(k) for k, v in pmap.items()}
print('Per-class info:')
for cls in sorted(set(y.tolist())):
    n = (y == cls).sum()
    orig = rev.get(cls, cls)
    # Decode pattern bits -> which raw channels active (ch 64..75)
    bits = [(orig >> i) & 1 for i in range(12)]
    active_ch = [64+i for i, b in enumerate(bits) if b]
    print(f'  cls {cls}: n={n} orig={orig} active_raw_ch={active_ch}')
