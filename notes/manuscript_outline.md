# Introduction

- description of influenza evolutionary mechanisms/patterns and available 'raw' data (without tree inference)
    - __figure__: HA strain succession

- Approaches for using HA sequence data for prediction towards seasonal vaccine development:
    - prediction of sequence evolution:
        - Neher, Russell, Shraiman (2014), eLife
        - Luksza and Laessig (2014), Nature
    - mapping and prediction of antigenic evolution:
        - Neher et al. (2016), PNAS
    - only directly useful for seasonal vaccine development, not for long-term protective approaches

- Approaches for development of universal, cross-protective vaccines
    - strategies that target conserved, but less accessible, sites (stem, receptor binding region)
    - key challenge: less accessible therefore hard to develop abs against and intrinsic fitness unknown (might be able to escape easily)

- approaches for inferring intrinsic fitness of other highly mutable viruses:
    - HIV, (polio) (papers by Chakraborty lab)
    - uses equilibrium approach with assumption that human immune response acts as random force, this assumption is not valid for influenza that is by population-wide immune memory driven out of equilibrium

- fitness inference method for time series data (MPL):
    - considers time-invariant fitness landscape (does not try to disentangle immune-dependent from intrinsic contribution)
    - does not take into account fitness coupling of pair mutations
    

# Model of influenza antigen evolution

- model mechanisms, assumptions, temporal coarse-graining (see model schematic in slides)
    - __figure?__: model schematic
- mathematical model formulation (see slides) (take out details for methods)
    - essential model equations
    - fitness formulation
    - __table__: model parameters plus inference parameters
    - __figure__: strain succession from simulated data 


# Inference of intrinsic fitness from flu-like simulated sequence data

- inference method and stringency assumption:
    - essential equations
    - refer to parameter table
    - __figure__: widths of fitness dists from example simulation
    - __figure?__: log(x(t+1)/xm(t)) for various B/N

- test of inference performance
    - __figure__: inferred vs. simulated fitness params with correlation (and classification curves) for one example analysis
    - __figure__: performance measure as function of sample size/ number of seasons
    - __figure__: performance and stringency measure as function of sequence length
    -__figure__: performance and stringency measure as function of population size

# Discussion 

