## Penn Neurotech Hack — Science Track

We chose \( N = 14 \) to keep the output space manageable. Larger alphabets made decoding unstable and harder to learn, while this size preserved enough expressivity for control.

Our decoder uses a windowed ridge regression approach. Neural data is binned at 10 ms resolution and a 20-bin (200 ms) causal context is flattened into a single feature vector. We compute both mean and RMS features, resulting in a 2560-dimensional input. RMS captures signal power that mean-only features suppress, improving validation correlation by ~0.05.

We standardize inputs and train ridge regression models for each output channel. Ridge was chosen for its stability under high-dimensional, correlated inputs and its suitability for real-time use. Sample weighting emphasizes behaviorally relevant periods so the model is not dominated by idle data. The regularization parameter \( \alpha \) is selected using a held-out validation file to balance bias and variance.

Predictions are clipped to valid controller ranges ([-1,1] for sticks, [0,1] for buttons) and rescaled to device units (int16). The full pipeline is causal and computationally efficient, making it suitable for live decoding.

### Pipeline
- 10 ms bins with 200 ms context (20 bins)  
- Mean + RMS features → 2560-dimensional input  
- Standardization → ridge regression → clipping → rescaling
- 
### Calibration Note
The calibration system (estimating optimal \( N \), threshold \( \theta \), and resulting bit rate \( B \)) is implemented. Although sweep results are collected and visualized, the selection criteria and downstream integration are not fully validated, so reported optimal values should be interpreted as approximate.

### What Didn’t Work
- Classification on discretized targets performed at chance (balanced accuracy 0.33 for 4-class, 0.07 for \( N = 14 \))  
- MLP regression (2-layer, 256/128, dropout 0.1) severely overfit (train corr ≈ 0.99, poor validation performance)  

---

## System Additions

### GameState
- Added `peak_streak: int = 0`, updated in `process_input` on each correct hit and reset in `reset()`

### CollectState
- Added `calib_curve_ns` and `calib_curve_bs` to store calibration sweep results  
- `poll()` now unpacks a 6-tuple from `calib_done`:  
  `(N, theta, B, path, curve_ns, curve_bs)`

### Calibration Worker
- `_auto_calib_worker` extracts `curve_ns` and `curve_bs` from the sweep results and includes them in the `calib_done` message

### Visualization
- `_draw_bn_chart` renders a native pygame bar chart:
  - Grey axes  
  - Blue bars for each tested `N`  
  - Green bar + “N*” label for the optimal value  
  - Bit rate `B` displayed inside each bar  
  - `N` labeled along the x-axis  

### Results Screen
- `draw_results_screen` provides a full-screen, two-column layout:
  - Left panel:
    - Bit rate (large)
    - Accuracy (%)
    - Total / correct / incorrect counts
    - Peak streak
    - Elapsed time
  - Right panel:
    - Optimal `N*`, `θ`, peak `B`
    - B(N) bar chart or fallback message if calibration is skipped
  - Bottom controls:
    - `R — PLAY AGAIN` (green)
    - `ESC — QUIT`

### Main Loop Behavior
- When `state.trial_ended` is `True` (outside intro/collect screens):
  - Renders results screen  
  - Continues polling calibration in case it finishes after the trial ends  
- Pressing `R` triggers `state.reset()`:
  - Sets `trial_ended = False`  
  - Hides results screen and restarts the game  

---

### How To Upload The Model And The App
1. Extract the `synapse-example-app` folder  
2. Run: `synapsectl apps build .`  
3. Run: `synapsectl -u "tamarin-of-silent-virtuosity" apps deploy .`  
4. Run: `synapsectl -u "tamarin-of-silent-virtuosity" deploy-model model.onnx --name model`  
5. Run: `synapsectl -u "tamarin-of-silent-virtuosity" start ./config/simulator_32ch.json`  

### How To Run The Game
- Execute `run.sh` to start the game
