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

- Approaches for development of universal, cross-protective influenza vaccines
    - strategies that target conserved, but less accessible, sites (stem, receptor binding region)
    - key challenge: less accessible therefore hard to develop abs against and intrinsic fitness unknown (might be able to escape easily)

- approaches for inferring intrinsic fitness of other highly mutable viruses:
    - HIV, (polio) (papers by Chakraborty lab)
    - use equilibrium approach with assumption that human immune response acts as random force, this assumption is not valid for influenza that is by population-wide immune memory driven out of equilibrium
    - Cocco and Monasson review for standard approaches (adaptive cluster expansion)

- fitness inference method for time series data (MPL):
    - considers time-invariant fitness landscape (does not try to disentangle immune-dependent from intrinsic contribution)
    - does not take into account fitness coupling of pair mutations
    

# Model of influenza antigen evolution

## Sequence representation
 - binary sequence representation 

## Fitness model
### Intrinsic fitness representation
- F_int = ...
- fields and couplings
- refer to HIV papers and others using this type of Ising fitness representation

### Representation of host immunity-mediated fitness cost
- F_host = ...
- refer to luksza et al. and others before

## Sequence selection
- x(S_j, t+1) = ... 

## Sequence mutation
- x_m = ...

# Analysis and fitness inference based on simulated sequence data

## Simulation of influenza antigen evolution
- __table__: model parameters plus inference parameters
- __figure__: strain succession from simulated data 

## Stringent selection regime
- __figure__: widths of fitness dists from example simulation
- __?figure__: log(x(t+1)/xm(t)) for various B/N
- observation that F is narrowly distributed
- ? observation that log(x/x') is very noisy due to subsampling

## Method for intrinsic fitness inference
- based on stringent selection assumption ...
- {h,J,F^*} = arg min...
- solution method M = ..., refer to Hastie and Tibshirani 2009, p. 44 ff

## Inferring the intrinsic fitness from simulated flu-like sequence data

- test of inference performance
- __figure__: inferred vs. simulated fitness params with correlation for one example analysis
- __figure__: classification curves for one example analysis
- __figure__: performance measure as function of sample size/ number of seasons
- __figure__: performance and measure as function of sequence length aand population size

# Discussion 

- summary of results, we were able to
    - infer intrinsic fitness landscape from influenza-like immune-driven sequence data
    - approach takes into account single and pair mutations
    - like MPL, controls for genetic linkage effects, by taking into account full sequence information for immune-dependent fitness estimate of each sample
    - simulation of influenza-like evolution allows to analyze performance under different conditions

- in comparison with previous studies (MPL, HIV studies):
    - were able to disentangle time-varying immune-dependent from intrinsic fitness in inference
    - were able to infer single and pairwise mutational effects

- outlook:
    - inference of intrinsic fitness from HA yearly sequence data
        - use HI assays for cross-immunity data
        - identify regions/sites/combos that are ost vulnerable
        - design vaccine or antibody treatment, which target those vulnerable regions
    - challenges
        - useful coarse-graining of HA data for fitness inference with the available data
        - test prediction with simulation/prediction on held-out test set/ predictions that can be verified in small-scale experiment
        - challenge of development of treatment/prevention strategy (vaccination, antibody treatment) based on intrinsic fitness information
        - will there be specific regions/sites/combinations of sites that are most vulnerable independent of the current sequence state and sequence background?
        - we don't want to/need to target epitopes that are already highly targeted by immune memory responses
        - we need an accurate cross-immunity measure for our model (based on Hamming distance? distance within each epitope?)
        - might have to take into account that different head epitopes are not equally targeted?
        might need to take into account epitope/site accessibility to not infer that instead of actual intrinsic fitness
- additional considerations for effective influenza treatments:
    - T-cell immunity
    - other antigens besides HA

# References for influenza evolution models, motivating our model description

- Yan, Neher and Shraiman 2019, eLife
    - Phylodynamic theory of persistence ,extinction and speciation of rapidly adapting pathogens
    - mapping multi-strain SIR model on traveling-wave model
    - find that the persistence of a rapidly evolving Red-Queen-like state typical for seasonal influenza requires long-ranged cross-immunity and sufficiently large population sizes
    The RQS is generally unstable towards speciation and extinction
    - influenza-like behavior: rapid evolution while maintaining limited genetic diversity, the most recent ancestor of the population in subtype A/H2N2 is rarely more than 3-5 years in the past [Rambaut et al. 2008]
    - lasting immunity against specific influenza strains [Fonville et al. 2014]
    - escape by accumulating amino acid substitutions in surface glycoproteins [Koel et al 2013, Wilson and Cox 1990]
    - within each subtype many HA sequence variants co-circulate [Rambaut et al. 2008, Fitch et al. 1997],
    differing by around 10 substitutions from eachother [Strelkowa and Laessing 2012]
    - decay of immune cross-reactivity over around 10 years [Smith et al. 2004, ...]
    - Epidemiological dynamics of influenza often modeled with generalizations of classic SIR model to multiple variants [Kermack and McKendrick 1927, Gog and Grenfell 2002], common approach has been to impose discrete one-dim strain space in which new strains are generated from adjacent strains and susceptibility reduces with distance in strain space [Andreasen et al. 1996; Gog and Grenfell 2002]; such models result naturally in traveling waves with pathogen pop. moving through strain space [Lin et al. 2003]
    - the antigencially evolving populations are related to general models of rapid adaptation with populations moving towards higher fitness [Tsimring et al. 1996, Rouzine et al. 2003, Desai and Fisher 2007, Neher et al. 2014, Neher 2013a] 
    - [Rouzine and Rozhnova 2018] describe explicit mapping between SIR model in 1-d antigenic space and traveling wave models (TW) in fitness
    - 1D TW models naturally result in spindly phylogenies (one possible direction for escape)
    - Influenza viruses have high-dimensional antigenic space [Perselson Oster 1979, Wilson and Cox 1990], such that different strains can follow different escape paths and can diverge sufficiently until they no longer compete for hosts and propagate independently afterwards
    - How is the spindly phylogeny maintained? Several computational studies have addressed this and identified cross-immunity [Bedford et al. 2012, Tria et al. 2005, Koelle et al. 2011, Ferguson et al. 2003, ...] as well as deleterious mutations [Koelle and Rasmussen 2015, Gog and Grenfell 2002] as critical parameters
    - cross-immunity expression (equivalent to our model)
    motivated by host-level susceptibility (Eq. 3) [Wikramaratna et al. 2015] and 'order one independence closure' or mean-field approximation [Kryazhimskiy et al. 2007, Weiss 1907, Landau and Lifshitz 2013] leading to exponential expression, analogous to our model Sa = exp(-sum Kab Rb) with Rb fraction of population recovered from strain (also analogous to models by [Luksza and Laessig 2014] and [Gog and Grenfell 2002])
    - authors follow [Luksza and Laessig 2014] for functional form of cross-immunity
    - HOW/WITH WHICH ASSUMPTIONS CAN I TRANSFORM THE GIVEN FORM OF INFECTION GROWTH INTO OUR MODEL WITH FINT AND FHOST? Check Luksza and Laessig who use an equivalent model to ours (see eq. 1 and 2, 11-14 in methods of their paper)
    - 

