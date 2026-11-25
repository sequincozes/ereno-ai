# Related work and differentiator (survey of 2026-10-22)

Survey carried out for this project. Each work was classified along three
pillars:

- **P1, adversarial system**: adversarial attack generation, evasion, GAN, RL, LLM agent or adversarial retraining.
- **P2, IEC 61850 domain**: GOOSE, SV, MMS or digital substation.
- **P3, IDS**: intrusion or anomaly detection.

The **primary** works cover all 3 pillars and the **secondary** ones cover 2. In the table, `~` marks a partial
pillar: the attacker exists, but is manual or acts only on the sample.

For each work, the last column says what this project has that the other work does not. The complete
architecture of this project, for reference:

1. an LLM attacker that produces only an intent (IntentSpec), deterministically compiled into an attack configuration;
2. generation at the **generator** level (ERENO), with 12 GOOSE attack families;
3. an ML IDS (Random Forest);
4. an LLM Defender agent with a plan validated against the evidence in the detection report;
5. **adversarial retraining in rounds**, in which attack and defense co-evolve (E3);
6. comparison against non-LLM baselines (random search and an open-loop ladder), with paired repetitions and CIs (E1).

A DOI marked with ✔ was confirmed on the publisher's page or in Crossref. The others are unconfirmed.

## Verdict

**No work found does nearly 100% of the proposal.** The survey confirmed four gaps that
support the novelty:

1. No work does **adversarial retraining of a GOOSE IDS**. Retraining in rounds only appears outside
   ICS (SIGMA, Rivas et al.). This supports E3 as the central result.
2. An **LLM attacker against an IDS in IEC 61850** only exists in the group's own SBSeg 2026. CritBench has
   an offensive LLM, but no IDS. GridStratLLM has an LLM and an IDS, but outside IEC 61850.
3. **Red×blue co-evolution with an LLM** only appears in a generic network (Du 2026) and without retraining the model.
4. **Generator-level perturbation** only appears in SBSeg 2026. The others perturb samples
   (Zaboli & Hong 2025) or design the attack by hand (Ayyalusamy 2026, Kamran 2026).

Two groups are converging on this space and need to be tracked: the one at KIT
(Hagenmeyer: CritBench, GridStratLLM, GOOSEIDRS) and the one of J. Hong (Zaboli, Cibin, SubCASP).

## Primary (P1 + P2 + P3)

| Work | DOI / ID | P1 | P2 | P3 | Sim. | What overlaps | Differentiator of this project |
|---|---|---|---|---|---|---|---|
| Quincozes, Mundt, Rankin, Quincozes, Severo, Mossé. *An LLM-Based Agentic Pipeline for Generator-Level Adversarial Evaluation of Smart Grid IDSs*. SBSeg 2026 | 10.5753/sbseg.2026.29283 ✔ | ✓ | ✓ | ✓ | high | LLM attacker at the ERENO generator level against a frozen RF | This is the base work. The differentiator is **closing the loop**: a validated LLM Defender, adversarial retraining in rounds (E3), 12 families instead of only masquerade, non-LLM baselines with repetitions and CIs, LLM output through tool calling and a deterministic compiler instead of a free-text parser (in SBSeg, gpt-oss-120b got the format right only 1 time in 30), and measured token cost. The limitations and future work of SBSeg ask for exactly non-LLM baselines and repetitions. |
| Zaboli, Hong. *Generative AI for Critical Infrastructure in Smart Grids: A Unified Framework for Synthetic Data Generation and Anomaly Detection*. 2025 | arXiv 2508.08593 | ~ | ✓ | ✓ | medium-high | Synthetic adversarial GOOSE (AATM technique) used to evaluate detectors in IEC 61850 | Here the attack is born in the **generator**, with protocol semantics, and not through gradient-based mutation of samples. There is an LLM attacker with an explicit objective, a Defender that retrains the IDS and a co-evolution cycle. AATM generates data only once; there is no arms race. |
| Ayyalusamy, Bala Krishnan, Xiao, Kandasamy, Abidi. *IEC 61850 GOOSE and Sampled-Values Dataset from Hybrid Digital Twin with Replay and Adversarial-Aware Attacks*. IEEE DataPort 2026 | 10.21227/xxq7-sk20 | ~ | ✓ | ✓ | medium | Adaptive GOOSE attacker (stNum increment and spoofed OUI) to test robustness against evasion | It is a static, non-peer-reviewed dataset, with a fixed attacker of manual rules. Here the attacker **searches** for the variant by itself and the defense reacts. This dataset can serve as an **external test** of the hardened IDS in ERENO. |

## Secondary (2 pillars)

| Work | DOI / ID | P1 | P2 | P3 | Sim. | What overlaps | Differentiator of this project |
|---|---|---|---|---|---|---|---|
| Martins, Quincozes, Siervo, Quincozes, Luizelli. *From Red Flags to Detection Rules: An LLM-driven Pipeline for Real-Time GOOSE Intrusion Detection and Prevention*. SBRC Estendido 2026 | 10.5753/sbrc_estendido.2026.23263 ✔ | | ✓ | ✓ | medium | Defender LLM over ERENO data | There, only the defensive side exists. Here there is an attacker that adapts to the defense, and the Defender's plan goes through an evidence gate against the detection report before being accepted. It is the precedent for the LLM Defender. |
| Keppler, Gstür, Hagenmeyer. *CritBench: A Framework for Evaluating Cybersecurity Capabilities of LLMs in IEC 61850 Digital Substation Environments*. 2026 | arXiv 2604.06019 (KIT cites 10.1145/3765611.3815141, unconfirmed) | ✓ | ✓ | | medium | LLM as an offensive agent in IEC 61850 (SCD, PCAP, live IED) | CritBench **measures the capability** of the LLM on offensive tasks. Here the attacker serves to **harden an ML IDS** in a cycle with the defense. CritBench has no IDS, defender or training dataset. |
| Kellerer, Hagenmeyer. *GridStratLLM: Agent Framework for Coordinated Cyberattacks on the Smart Grid with LLMs*. ACM e-Energy 2026 | 10.1145/3765611.3815147 ✔ | ✓ | | ✓ | medium | LLM with tool calling orchestrating attacks and an HIL dataset to train an IDS | There the protocols are S7 and Modbus, not IEC 61850, and the dataset is generated once. Here it is GOOSE, the IDS is retrained against the attacker in rounds, and the LLM never touches the runtime, because the deterministic compiler sits in the middle (GridStratLLM executes malware). |
| Du. *GenAI-Powered Autonomous Cyber Offense-Defense: An Explainable LLM Red-vs-Blue Simulation and Self-Defense Framework*. J. Cyber Security 2026 | 10.32604/jcs.2026.075976 | ✓ | | ✓ | medium | LLM Red and Blue in co-adaptation, with IDS and SIEM | There the Blue's adaptation is through **prompt and rules**, in a generic network. Here the defense **retrains the detection model**, in the GOOSE domain, and the effect is measured statistically (D_i × A_j matrix, no-defense control). |
| Rivas, Saika, Bakht, Piplai, Bastian, Shah. *Adapting Under Fire: Multi-Agent Reinforcement Learning for Adversarial Drift in Network Security*. SECRYPT 2025 | arXiv 2506.06565 | ✓ | | ✓ | medium | Red×blue arms race against a NIDS | There the agents are RL-based and perturb packets of a generic network. Here the attacker is an LLM, generation happens in the IEC 61850 generator and the defense is evaluated against non-LLM baselines. |
| Msika, Quintero, Khomh. *SIGMA: Strengthening IDS with GAN and Metaheuristics Attacks*. 2019 | arXiv 1912.09303 | ✓ | | ✓ | medium | Loop of generating an attack, measuring and retraining the IDS in rounds | It is the non-LLM analogue of E3. There the attack is per sample (GAN and metaheuristic), outside ICS. Here generation is semantic, in the GOOSE generator, guided by intent, with the LLM Defender and experimental controls (placebo and no-defense control). |
| Conceição Alberto, Sánchez, Dembele, Diop, Elbez, Hagenmeyer. *GOOSEIDRS: Intrusion Detection and Automated Response for IEC61850 GOOSE Masquerade Attacks*. ACM e-Energy 2026 | 10.1145/3765611.3815139 ✔ | | ✓ | ✓ | low | Masquerade IDS with stNum/sqNum features and data augmentation | There is no optimized attacker; the data augmentation is not adversarial. Here the IDS is tested and retrained against actively searched variants. |
| Zaboli, Choi, Song, Hong. *ChatGPT and Other Large Language Models for Cybersecurity of Smart Grid Applications*. 2023/2024 | arXiv 2311.05462 (the 10688863 in the spreadsheet is an IEEE number, not a DOI) | | ✓ | ✓ | low | LLM over GOOSE and SV | There the LLM is the **detector**. Here the LLM generates the attack and the defense plan, and detection remains an auditable ML model. |
| Moore. *Anomaly Detection in IEC-61850 GOOSE Networks: Evaluating Unsupervised and Temporal Learning for Real-Time Intrusion Detection*. 2026 | arXiv 2604.14233 | | ✓ | ✓ | low | Uses ERENO, with RF, autoencoders and GRU | There is no adversarial side: ERENO is used as a fixed dataset. Here ERENO is the **generator** that the attacker controls. |
| Cibin, Mulder, Carstens, Hong, Ştefanov. *Cyber Attacks Detection, Prevention, and Source Localization in Digital Substation Communication using Hybrid Statistical-Deep Learning*. 2025 | arXiv 2507.00522 | | ✓ | ✓ | low | IDS/IPS for SV | There, evasion appears only as future work. Here evasion is the object of study. |
| Herath, Liu, Hong, Park. *Strategy Phasing of Cyber Attacks on Digital Substations* (SubCASP). 2026 | arXiv 2607.27661 | ~ | ✓ | ✓ | low | Attack phases in a substation from IDS logs | There an HMM infers the phase of attacks that have already occurred. Here the attack variants are generated and the detector is hardened against them. |
| Kamran. *Stealthy GOOSE Data Generation and Unsupervised Intrusion Detection in IEC-61850 Substations*. Dissertation, KTH 2026 | no DOI (probably in DiVA) | ~ | ✓ | ✓ | low | Stealthy GOOSE at 3 levels and unsupervised IDS | There the stealth levels are designed by hand on a hardware testbed. Here stealth is **searched** automatically, with the detector reacting. His data can also serve as an external test. |
| Quincozes, Albuquerque, Passos, Mossé. *A survey on intrusion detection and prevention systems in digital substations*. Computer Networks 2021 | 10.1016/j.comnet.2020.107679 ✔ | | ✓ | ✓ | low | Survey of IDPS in substations | Cited as background reference for the domain. |

## Discarded (do not cover 2 pillars in a relevant way)

- **ICSSPulse** (arXiv 2602.20663): the LLM only writes Modbus/OPC UA pentest reports. No IDS and no IEC 61850.
- **ACEA** (arXiv 2609.08256): red×blue between LLMs on the security of the LLM itself (secret leakage). No network, no IDS and no retraining.
- **ROAM** (arXiv 2610.12730): attacker×defender RL as a moving target defense for PMUs (IEEE C37.118). It is neither IEC 61850 nor a retrained ML IDS.

## Open items

- The works in the spreadsheet `AGENTE + SMART GRID - LAB GRUPO 17.xlsx` **have not been re-verified**: Sen
  2024, Teryak 2023, Asimopoulos 2023, ADVIS-G, IDSGAN, attackGAN, TANTRA, Apruzzese 2022, the FDIA ones
  with GAN/RL, Mutua 2025 and Sensors 21(4):1554. Their DOIs still need to be checked.
- Several works were classified from the abstract only (SBSeg, Zaboli 2023, SubCASP, Martins).
- Check whether Zaboli & Hong 2508.08593 has been published in a journal and whether the final version retrains the detector.
