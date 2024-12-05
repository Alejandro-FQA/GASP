# 1D Quantum Harmonic Oscillator with Stochastic Reconfiguration and Neural Quantum States

In this repository we provide a simple PyTorch implementation to solve the 1D quantum harmonic oscillator (HO) with the stochastic reconfiguration method (SR) and two different types of complex-valued neural networks (NNs) employed as avefunction ansätze, or neural quantum states (NQS).

The first network consists of a single neuron, or perceptron, with a Gaussian activation function (GASP).
The second one, is a common multilayer perceptron or NQS.
In both cases, we use SR to obtain the ground state and perform subsequent dynamics upon a displacement of the trapping potential.

```
├── LaTeX                               # LaTex files
│   ├── GASP.pdf                            > Report
├── Mathematica                         # SR calculations for GASP
│   ├── GASP_2.nb                           > Two complex parameters
│   └── GASP.nb                             > One complex parameter
├── Python                              # Pytorch code
│   ├── analysis.py                         > data processing methods
│   ├── integrators.py                      > RK4 and Euler integrators
│   ├── main.py                             > main code
│   ├── models.py                           > GASP and NQS classes
│   ├── notes.txt                           
│   ├── parameters.py                       > list of parameters
│   ├── plots.py                            > plotting methods
│   ├── stochastic_reconfiguration.py       > SR methods
│   └── utilities.py                        > useful methods
├── .gitignore
├── environment_simple.yml
├── environment.yml                     # environment dependencies
├── LICENSE
└── README.md 
```