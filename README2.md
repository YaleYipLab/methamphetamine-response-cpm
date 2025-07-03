


A shared spatial topography links the functional connectome correlates of cocaine use disorder and dopamine D2/3 receptor densities.
================
------------------------------------------------------------------------

## contents

- [background](#background)
- [reference](#reference)
- [code release](#code-release)
- [data](#data)
- [atlas](#atlas)
- [questions](#questions)

------------------------------------------------------------------------

## background


------------------------------------------------------------------------

## reference

For usage of the ***manuscript***, please cite:
- **Ricard J.A.**, Labache L., Segal A., Dhamala E., Cocuzza C.V., Jones G., Yip S., Chopra S., Holmes A.J. (2023)
  A shared spatial topography links the functional connectome correlates of cocaine use disorder and dopamine D2/3 receptor densities.
  bioRxiv 2023.11.17.567591; doi: [https://doi.org/10.1101/2023.11.17.56759](https://doi.org/10.1101/2023.11.17.56759)

------------------------------------------------------------------------

## code release


All analyses were performed using *R version 4.1.0 (2021-05-18)* and *Python 3.8.8*.

1) network_based_stats_schaefertian.R - returns nbs_result05_covars_SchaeferTian_230421.rds result.
2)  visualization_nbs_output.R - creates "degree05neg_covars" and "degree05pos_covars" and "avg_masked_Negative_covars_joce.csv" & "avg_masked_Positive_covars_joce.csv" for the nbs05 network - to plot edges (plot classified edges) and create nbs network brains.
3)  receptorspins_240428.ipynb - runs neurotransmitter receptor density spearman’s ρ correlations after 10k spin test and random shuffling per hemisphere for subcortical regions against NBS output.
4)  generate_plots.Rmd - generates network plots for visualization

------------------------------------------------------------------------

## data

All the brain fMRI data are from the SUDMEX-CONN dataset.
(Angeles-Valdez, Diego, et al. "The Mexican magnetic resonance imaging dataset of patients with cocaine use disorder: SUDMEX CONN." Scientific data 9.1 (2022). You can find more on their github repo here: https://github.com/psilantrolab/SUDMEX_CONN

Receptor density data are from neuromaps.
(Markello, Hensen et al., "neuromaps: structural and functional interpretation of brain maps" Nature Methods (2022). You can find more here: (https://www.nature.com/articles/s41592-022-01625-w).


------------------------------------------------------------------------

## atlas

The atlas used in te paper are available in the `atlas` folder. This
folder contains combined: `Schaefer-Cortical` and `Tian-Subcortical` atlases.

- **Schaefer-Cortical** provide an atlas in standardized MNI volume space...
Full description of the atlas can be found here:
  [Schaefer-Cortical](https://academic.oup.com/cercor/article/28/9/3095/3978804). 

- **Tian-Subcortical** is a functional brain ROIs atlas (32 brain regions) based
  on resting-state fMRI data acquired in more than 1000 individuals. Full description of the atlas can be found there:
  [Tian-Subcortical](https://www.nature.com/articles/s41593-020-00711-6).


------------------------------------------------------------------------

## questions

Please contact me (Jocelyn A. Ricard) at ricard [at] stanford [dot] edu
