# BACQ_CNRS_D3.1
Code library for hardware agnostic energy optimizer and energetic analysis

Supporting files
1. circ_gen.py : Ansatz generation routines for HVA and RYA
2. custom_qpu.py : Custom QPU based on GPU with linear algebra backend using Cudapy
3. opto_gauss : Custom global depolarizing noise model called GaussianNoise and scipy based optimizer called Opto

Main files
1. energetics.py : EnergeticAnalysis class for obtaining algorithmic energy consumption, hardware specific energy consumption, and energetic efficiency.
2. VQE_energetics_demo.py : Example use of the EnergeticAnalysis class with a sample VQE problem: Heisenberg model.
3. har_agnostic_resource_optim_VQE: Hardware agnostic resource optimizer for VQE. Provides a set of parameters: N_g and epsilon to meet a user specified metric with minimal energy consumption.
