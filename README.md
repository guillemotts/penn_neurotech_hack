## Penn Neurotech Hack — Science Track

We chose N = 14, as we thought it would be too difficult to learn an alphabet any bigger than that. 
We use a windowed ridge regression decoder: neural data from the past N bins is flattened, standardized, and mapped linearly to controller outputs. Ridge is robust to noise and multicollinearity, which is critical for high-dimensional neural inputs.
Sample weighting emphasizes behaviorally relevant periods so the model does not get dominated by idle data. This improves performance where it matters.
We select the regularization parameter alpha using a held-out validation file to control overfitting and ensure generalization across recordings.
Outputs are clipped to valid controller ranges and rescaled to match device units, ensuring physically valid predictions at inference time.
The entire pipeline is causal and efficient, making it suitable for real-time decoding.

### How To Upload The Model And The App
1. Extract the zipped folder synapse-example-app.
2. synapsectl apps build .
3. synapsectl -u "tamarin-of-silent-virtuosity" apps deploy .
4. synapsectl -u "tamarin-of-silent-virtuosity" deploy-model model.onnx --name model
5. synapsectl -u "tamarin-of-silent-virtuosity" start ./config/simulator_32ch.json

### How To Run The Game
1. Run run.sh to start the game.
