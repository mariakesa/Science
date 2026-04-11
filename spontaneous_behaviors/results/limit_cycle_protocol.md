Yes — here is a clean summary of the protocol you used to obtain the loop-like attractor.

The protocol starts with aligned neural activity and behavior labels over time. At each timepoint, you have a neural population vector and a MoSeq syllable label. The first step is to standardize the neural activity and reduce it with PCA, keeping a fixed number of neural PCs. In your code, the scaler and PCA are fit on the first 75% of the recording and then applied to the remaining 25%, so the representation is learned from the training segment and transferred to the future segment.

Next, you convert the continuous neural PC trajectory into a one-step prediction dataset. For each timepoint (t), the model input is the current neural PC state (x_t) together with the current syllable (s_t), and the prediction target is the next neural PC state (x_{t+1}). The syllable labels are remapped to integer indices so they can be passed through an embedding layer.

The predictor itself has three parts. First, the current neural PC vector is passed through an MLP encoder to produce a neural-state feature (h_t). Second, the current syllable is passed through an embedding table to produce a behavior feature (e_t). Third, these two are concatenated and passed through a fusion MLP to produce a fused latent representation (z_t), and then a final linear layer maps (z_t) to the predicted next neural PC state (\hat x_{t+1}).

Training is done with teacher forcing on the training transitions only. That means the model always receives the true current neural PC state (x_t) and the true syllable (s_t), and it is optimized with mean squared error to match the true next neural PC state (x_{t+1}). During training, a teacher-forced loss is also computed on the held-out future segment as a diagnostic.

The attractor does not come from teacher forcing. It comes from a second evaluation phase: free-running autoregressive rollout on the held-out segment. In that phase, the model is initialized with a single starting neural PC state (x_0) from the beginning of the held-out transitions. Then, at each future step, it receives the current syllable (s_t) but no longer receives the true neural state. Instead, it feeds its own previous prediction back in as input. So the recurrence is induced through the predicted PC state:
[
\hat x_{t+1} = F(\hat x_t, s_t).
]
At each step, you also save the fused latent vector (z_t). This latent sequence is then projected to 2D with PCA and plotted through time. That free-running latent trajectory is what produced the loop-like structure.

So the attractor protocol, in one sentence, is:

> fit a one-step neural-PC predictor conditioned on movement syllables, then initialize it on the held-out future segment and let it roll forward on its own predicted states while recording the fused latent trajectory.

There is one subtle methodological caveat in the exact code you used. Because you concatenate train and test PCs and then split transitions at 75%, the first “test” rollout state is effectively at the train/test boundary rather than strictly inside the held-out block. So the attractor you got is best described as a rollout from the boundary into the future segment, not a perfectly pure test-only rollout. The overall idea is still sound, but when you turn this into a reusable library function, I would make the split explicit so that train and test transitions are built separately.

