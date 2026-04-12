import numpy as np, sys, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, 'src')
from penn_nh.decoder.linear_probe import LinearProbeFactory

N_CLASSES = 14
C_IN = 128
T_IN = 8

rng = np.random.RandomState(0)
M = N_CLASSES * 20
X = rng.randn(M, C_IN, T_IN).astype(np.float32)
y = np.tile(np.arange(N_CLASSES), 20)
fitted = LinearProbeFactory(C=1.0, max_iter=500).fit(X, y, n=N_CLASSES)

scaler = fitted.pipeline.named_steps['scaler']
lr = fitted.pipeline.named_steps['lr']
mean = torch.from_numpy(scaler.mean_.astype(np.float32))
scale = torch.from_numpy(scaler.scale_.astype(np.float32))
coef = torch.from_numpy(lr.coef_.astype(np.float32))
bias = torch.from_numpy(lr.intercept_.astype(np.float32))

class Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.register_buffer('coef', coef)
        self.register_buffer('bias', bias)
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = (x - self.mean) / self.scale
        logits = x @ self.coef.T + self.bias
        return torch.log_softmax(logits, dim=-1)

model = Stub().eval()
out = Path('models/decoder_stub.onnx')
out.parent.mkdir(parents=True, exist_ok=True)
dummy = torch.randn(1, C_IN, T_IN)
torch.onnx.export(model, dummy, out,
    input_names=['features'], output_names=['log_probs'],
    dynamic_axes={'features': {0: 'batch'}, 'log_probs': {0: 'batch'}},
    opset_version=17)
print(f'SAVED {out} input=(B,{C_IN},{T_IN}) output=(B,{N_CLASSES})')

# Verify
import onnxruntime as ort
sess = ort.InferenceSession(str(out))
test = np.random.randn(3, C_IN, T_IN).astype(np.float32)
onnx_out = sess.run(None, {'features': test})[0]
torch_out = model(torch.from_numpy(test)).detach().numpy()
assert np.allclose(onnx_out, torch_out, atol=1e-5)
print(f'ROUNDTRIP OK shape={onnx_out.shape}')
