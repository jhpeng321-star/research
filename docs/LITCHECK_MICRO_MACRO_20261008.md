# Literature check: micro-macro coupling for a learned traffic WM (2026-10-08)

Coordinator subagent (web search; items marked [mem] were recalled, not checked online). Verify venues/pages before submission.

## Closest prior work: cite and differentiate
1. **Xue, Vu, Dantsuji & Ngoduy (2026), TR-C 192:105914, "MPIDL"** (https://www.sciencedirect.com/science/article/pii/S0968090X26004006).
   - What they do: an LSTM predicts per-vehicle accelerations; a neural network estimates macro fields regularised by ARZ; a cross-scale loss ties the two.
   - Difference from ours:
     - their macro consistency is a training loss only, not a coupling applied at every rollout step;
     - their macro field is a fitted network, not a CTM run forward from t0;
     - their horizons are short multi-step prediction, not 300 s.
2. **Son, Qiao, Sewall & Lin (2022), ACM TOG 41(6)** (arXiv 2210.08046).
   - What they do: a differentiable hybrid simulator (ARZ macro + IDM micro).
   - Difference from ours: the scales are coupled at lane/region interfaces; there is no learned residual and no error control.
3. **Zhang & Redmill (2026), arXiv 2601.22242.**
   - What they do: align learned driving policies with aggregate macro statistics.
   - Difference from ours: the alignment comes through the loss, not through coupling during the rollout.

## Background to cite
- **Interface hybrids:** Bourrel & Lesort 2003 (TRR 1852); Leclercq 2007 (TR-B) [mem]; Moutari & Rascle 2007 [mem]; NoDEA 2017; Open Traffic Models arXiv 1908.04009.
- **Region hybrids:** Sewall, Wilkie & Lin 2011 (ACM TOG).
- **PIDL:** Shi, Mo, Huang, Di & Du (IEEE T-ITS 2022); Shi, Mo & Di (AAAI 2021); Mo, Shi & Di (TR-C 2021).
- **IDM + learned residual:** PERL (arXiv 2309.15284).
- **Micro -> macro direction:** nonlocal-LWR PINN (arXiv 2506.18357).
- **Compounding error over 30-300 s:** SceneDiffuser++ (arXiv 2506.21976).
- **Differentiable LWR/LTM:** UNsim (arXiv 2604.11380; argues hard min/max needs no relaxation - be ready to justify the soft-min); Reilly et al. 2015 (adjoint LWR).
- **Agent-continuum analogues:** Narain et al. 2009 (crowds) [mem]; Learning Hybrid Particle-Continuum models (OpenReview); MF-MDP (arXiv 2604.05516).
- **Nudging:** von Storch et al. 2000 (spectral nudging) [mem] - our nudge is Newtonian relaxation; acknowledge it.

## Novelty assessment
- **Appears new:**
  - an independent macro CTM run from t0 (stop-gradient) and coupled to a learned micro WM at every rollout step;
  - a learned, horizon-dependent nudge weight;
  - end-to-end training through a soft-min CTM with soft kernel binning;
  - the target: phantom/missed jams over 10-300 s for CAV-controller evaluation;
  - merge capacity drop in the macro prior.
- **Risks:**
  - the idea can be framed as nudging / data assimilation toward a coarse forecast, so the contribution must rest on the learned weight, end-to-end training and the traffic results;
  - an equilibrium-FD macro model may wash out real stop-and-go waves; we must show the coupling does not erase real oscillations or CAV-controller effects.
