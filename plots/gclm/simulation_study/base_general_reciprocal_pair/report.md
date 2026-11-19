# Reciprocal pairs and mixing in the base general-graph run

Five independent G2, n=800, λP=5 chains were analyzed from their 500,000 retained draws each. All positions below are zero-based *retained* indices, with half-open intervals; vertices in figures and tables are one-based. The samples were not regenerated. The stored log-likelihood is the total log-likelihood up to a constant.

## Main findings

Across 2,500,000 retained draws, the mean reciprocal-pair count is 0.316; 27.8% of draws contain at least one reciprocal pair. The pooled correlation of k and c(G) is 0.626. Bulk ESS for c(G) is 110.7 and rank-normalized split R-hat is 1.042. These are diagnostic estimates, not independent sample counts.

The most frequent reciprocal pairs are 2↔4 (10.18%), 1↔2 (9.11%), 3↔4 (4.18%), 1↔5 (3.77%). See `reciprocal_pair_heatmap.pdf` and `edge_and_pair_inclusion.csv`.

First-chain exploratory windows (these were supplied for inspection, not inferred change points):

| Retained indices | Mean k | Mean c(G) | P(c>0) | Mean η | Mean total log-likelihood |
|---|---:|---:|---:|---:|---:|
| 0–60,000 | 2.866 | 0.170 | 0.158 | 0.902 | -196.680 |
| 60,000–130,000 | 2.589 | 0.104 | 0.099 | 0.954 | -197.056 |
| 130,000–240,000 | 2.729 | 0.127 | 0.120 | 0.924 | -195.551 |
| 240,000–270,000 | 2.898 | 0.144 | 0.136 | 0.865 | -198.514 |
| 270,000–330,000 | 3.690 | 0.818 | 0.745 | 0.889 | -194.512 |
| 330,000–500,000 | 2.966 | 0.220 | 0.207 | 0.893 | -194.821 |

In chain 1's 270,000–330,000 interval, 1↔2 occurs in 72.6% of draws. Its longest uninterrupted residence is [287,859, 331,078), or 43,219 draws, overlapping the detected high-k stretch [288,846, 329,922), where it occurs in 100.0% of draws. The detailed graphs and pair frequencies are in `regime_top_graphs.csv` and `regime_pairs.csv`. See `trace_chain_01.pdf` and `focus_240000_350000.pdf`.

The other chains were screened with a 5,000-draw rolling mean of k exceeding that chain's median k by 0.8, merging gaps of at most 1,000 draws and keeping stretches of at least 5,000 draws. This is an exploratory rule, not a formal change-point procedure. The longest such stretches outside chain 1 are: chain_02 [82,490, 171,821) (89,331 draws; 1↔5 present 99.3% of draws; mean η=1.219); chain_05 [330,313, 406,918) (76,605 draws; 2↔4 present 100.0% of draws; mean η=0.912); chain_05 [0, 64,443) (64,443 draws; 2↔4 present 97.6% of draws; mean η=0.923). The five-chain trace figure plots every 25th saved draw for display only. See `trace_all_chains.pdf` and `automatic_high_k_regimes.csv`.

## Transitions and continuous parameters

The saved rows record the selected move and whether it was accepted, but not the identity of a rejected proposed edge or its proposed parameter values. Thus `move_acceptance.csv` gives overall birth/death proposal acceptance; it cannot give an acceptance rate *conditional on attempting to complete or break a particular pair*. `pair_transition_residence.csv` counts only observed, accepted pair completions/breaks and includes residence durations. For the most frequent pair 2↔4, the longest observed continuous residence is 77,488 draws in chain_05. Across all retained draws, 7.5% of birth proposals and 41.5% of death proposals were accepted. Chain 1 nevertheless has 746 observed completions and 746 breaks of 1↔2; its median residence is 19 draws against a maximum of 43,219. Many short visits coexist with rare long ones. `transitions_near_regime_boundaries.csv` lists pair transitions within 2,000 draws of selected regime boundaries.

When c(G)=0 versus c(G)≥1, the pooled *conditional mean* drift matrices differ by 0.435 in Frobenius norm, and the pooled conditional mean stationary covariance matrices by 0.015. Their mean η values are 0.899 and 0.942; mean total log-likelihoods are -195.224 and -193.697. These summaries mix over different graphs and chains, so small mean-matrix distances do not establish draw-wise observational equivalence. The group matrices and the most common pair's comparison are in `conditional_mean_matrices.csv` and `matrix_group_comparisons.csv`. Conditional η and likelihood distributions are shown in `continuous_parameters_by_c.pdf`; their quantiles are in `continuous_by_reciprocal_count.csv`.

Pooled η quartile cutoffs are 0.8687 and 0.9349. `eta_quartile_top_graphs.csv` lists common graphs in low- and high-η draws for each chain; these are descriptive partitions of autocorrelated samples.

## Interpretation and limits

Reciprocal pairs coincide with the detected elevated-k stretches, often with one pair dominating an entire stretch. This supplies empirical motivation to test a simultaneous cycle birth/death proposal. Frequent pair entries and exits mean these results do not support a universal one-edge transition barrier. They also do **not** establish that the intermediate one-edge graph is an acceptance barrier: rejected target edges and their likelihoods were not stored. Similar likelihoods across structural groups also do not by themselves prove structural non-identifiability. Slow mixing, posterior multimodality, and observational equivalence remain distinct explanations. The between-chain discrepancies and low ESS mean pooled frequencies should be interpreted cautiously.

## Reproduction

From the repository root, run `python3 plots/gclm/simulation_study/base_general_reciprocal_pair/analyze_reciprocal_pairs.py`. No simulation output is changed. All figures are PDFs; all numerical summaries are CSVs.
