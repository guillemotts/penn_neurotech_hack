## Penn Neurotech Hack — Science Track

We chose \( N = 14 \) to keep the output space manageable. Larger alphabets made decoding unstable and harder to learn, while this size preserved enough expressivity for control.

Our decoder uses a windowed ridge regression approach. Neural data is binned at 10 ms resolution and a 20-bin (200 ms) causal context is flattened into a single feature vector. We compute both mean and RMS features, resulting in a 2560-dimensional input. RMS captures signal power that mean-only features suppress, improving validation correlation by ~0.05.

We standardize inputs and train ridge regression models for each output channel. Ridge was chosen for its stability under high-dimensional, correlated inputs and its suitability for real-time use. Sample weighting emphasizes behaviorally relevant periods so the model is not dominated by idle data. The regularization parameter \( \alpha \) is selected using a held-out validation file to balance bias and variance.

Predictions are clipped to valid controller ranges ([-1,1] for sticks, [0,1] for buttons) and rescaled to device units (int16). The full pipeline is causal and computationally efficient, making it suitable for live decoding.

### Pipeline
- 10 ms bins with 200 ms context (20 bins)  
- Mean + RMS features → 2560-dimensional input  
- Standardization → ridge regression → clipping → rescaling  

### What Didn’t Work
- Classification on discretized targets performed at chance (balanced accuracy 0.33 for 4-class, 0.07 for \( N = 14 \))  
- MLP regression (2-layer, 256/128, dropout 0.1) severely overfit (train corr ≈ 0.99, poor validation performance)  

### How To Upload The Model And The App
1. Extract the `synapse-example-app` folder  
2. Run: `synapsectl apps build .`  
3. Run: `synapsectl -u "tamarin-of-silent-virtuosity" apps deploy .`  
4. Run: `synapsectl -u "tamarin-of-silent-virtuosity" deploy-model model.onnx --name model`  
5. Run: `synapsectl -u "tamarin-of-silent-virtuosity" start ./config/simulator_32ch.json`  

### How To Run The Game
- Execute `run.sh` to start the game
