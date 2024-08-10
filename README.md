# Autoencoder

To detect the extremely high energetic (EHE) neutrinos and track where they come from
and find out what might have created them an enormous detector is required. The method
of detecting radio signals, created when the neutrinos interact with the antarctic ice makes
it possible to create these enormous detectors at a reasonable price as in the ”Antarctic
Ross Ice-shelf ANtenna Neutrino Array” (ARIANNA) project [3].
The radio signal is generated through the Askaryan effect, and it is possible to calculate
Askaryan radiation theoretically. There are experimental measurement of the Askaryan
effect that are in agreement with the theoretical predictions [4].
The energy from the Askaryan radiation is low, so the sensors have to be as sensitive as
possible. However, this also generates a lot of thermal noise. This leads to a lot of data
that has to be analyzed. In a previous pilot study in the ARIANNA project, a neural
network was used to distinguish noise from signal. The neural network was trained on
simulated data, including signals and noise, to classify them [2].
There is always a hazard in relying on simulated data, since no high-energy neutrino
has been observed yet. Instead of using a neural network that has been trained on a
classification task, one can use an autoencoder that only trains on noise. The idea is that
the autoencoder should then detect anomalies, or signals, in the real data.
The purpose of this project is to develop a method that can detect signals in a dataset
that contains both signals and noise. By using an autoencoder, a type of neural network,
that is trained only on noise but can distinguish between signals and noise when exposed
to both, we aim to determine whether a given input is a signal or not.