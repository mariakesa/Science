In the unpermuted case, the latent is not being determined mainly by the current syllable alone. Instead, it looks like the model is in a regime where

zt=fusion(ht,et)

reflects a structured interaction between:

the current autoregressive state carried by the predicted PCs
the current syllable embedding
the fact that this syllable arrives in the “right” temporal context

neurobehavioral loop
movement-coupled neural loop
behavior-conditioned neural limit-cycle-like manifold

We do not see the loop under teacher forcing because teacher forcing injects the true held-out neural state at every step, preserving the full heterogeneity of the data. The loop emerges only in free-running rollout, where the model evolves according to its own reduced learned dynamics and can collapse onto a simpler self-consistent attractor.