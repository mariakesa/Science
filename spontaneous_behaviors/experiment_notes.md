Data: https://zenodo.org/records/17488068
Paper: https://www.cell.com/neuron/pdf/S0896-6273(25)00894-3.pdf

Initial chat: chatgpt.com/g/g-p-67b49a58bff88191838dd62962204448-neuraldatascience-mitneuralcomputation/c/69d91294-3ee8-8387-9289-b5d15b606435

Idea: Fit CEBRA to neural data with MoSeq syllable column in behavior
as an auxiliary variable. Roll own implementation of CEBRA and
augment the algorithm with novelty.

Include Takens.

The idea is to recover the phase space of spontaneous behavior from
neural and behavioral data. 

The most ambitious version:
a shared latent dynamical space in which neural and behavioral trajectories are different projections of the same underlying process.

Heavy inspiration from Nonlinear dynamics. Watch this course: https://www.youtube.com/@jasonbramburger/videos

Experiment idea: Can we parse neural activity in the same way
we parse sounds in speech recognition? And how do these neural
syllables align with behavioral syllables?--> This is doesn't work because
the data is z-scored binned spikes.

