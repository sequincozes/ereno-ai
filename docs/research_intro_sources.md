# References for the ERENO-AI introduction

Checked on September 17, 2026. The statements below are limited to the abstracts and metadata of the primary sources indicated. They do not constitute a systematic review nor demonstrate that the proposed architecture is novel.

## ERENO

Silvio E. Quincozes, Célio Albuquerque, Diego G. Passos and Daniel Mossé. **ERENO: A Framework for Generating Realistic IEC–61850 Intrusion Detection Datasets for Smart Grids**. IEEE Transactions on Dependable and Secure Computing, 21(4), 3851–3865, 2024. **DOI: 10.1109/TDSC.2023.3336857**.

Primary source: [IEEE record](https://ieeexplore.ieee.org/document/10339874/). The indexed text of the publisher's page reports early access publication on December 4, 2023 and the final issue of July/August 2024; use 2024 in the reference with volume and pages. The DOI and issue year also appear in [Célio Albuquerque's ORCID](https://orcid.org/0000-0002-7959-6569).

Supported claims: ERENO generates synthetic traffic according to IEC 61850 specifications and provides datasets with normal traffic and seven attack classes. The paper identifies the lack of representative data and the use of proprietary datasets as obstacles to evaluating and reproducing IDS results in substations. Do not attribute to the original ERENO an LLM-based multi-agent cycle or results of the ERENO-AI proposal.

## GOOSE attacks

Taha Selim Ustun, Shaik Mullapathi Farooq and S. M. Suhail Hussain. **A Novel Approach for Mitigation of Replay and Masquerade Attacks in Smartgrids Using IEC 61850 Standard**. IEEE Access, 7, 156044–156053, 2019. **DOI: 10.1109/ACCESS.2019.2948117**.

Primary source: [institutional record with abstract and metadata](https://pure.kfupm.edu.sa/en/publications/a-novel-approach-for-mitigation-of-replay-and-masquerade-attacks-/). IEEE identifier: 8873588. [DOI](https://doi.org/10.1109/ACCESS.2019.2948117).

Supported claims: GOOSE messages can be subject to replay and masquerade; altering the content makes it harder to tell legitimate messages from forged ones. The paper analyzes these attacks and implements an authentication and integrity solution, evaluated in a laboratory. Avoid stating that every current IEC 61850 implementation is unprotected or that the study proves physical consequences in any installation.

## Detection in substations

Silvio E. Quincozes, Célio Albuquerque, Diego Passos and Daniel Mossé. **A survey on intrusion detection and prevention systems in digital substations**. Computer Networks, 184, 107679, 2021. **DOI: 10.1016/j.comnet.2020.107679**.

Primary sources: [Elsevier publisher](https://www.sciencedirect.com/science/article/pii/S1389128620312895), [copy on the author's institutional page](https://www.facom.ufu.br/~sequincozes/producoes/quincozes2021survey.pdf).

Supported claims: digital communication between substation devices enables monitoring and control; timing requirements and operational characteristics make the automatic transposition of conventional IDS inadequate. The study organizes IDS design and evaluation aspects and identifies limitations in the face of masquerade. The characterization of the state of the art is from 2021; do not present its assessment as a survey updated through 2026.

## Complementary notes verified by the main agent

B. Biggio, G. Fumera and F. Roli. **Security Evaluation of Pattern Classifiers under Attack**. IEEE Transactions on Knowledge and Data Engineering, 26(4), 984–996, 2014. **DOI: 10.1109/TKDE.2013.57**. Primary source: [IEEE](https://ieeexplore.ieee.org/document/6494573). It underpins the evaluation of classifiers under intentional manipulation and adversarial conditions. It does not prove the specific effectiveness of ERENO-AI.

Qingyun Wu et al. **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation**. arXiv preprint, 2023. **DOI: 10.48550/arXiv.2308.08155**. Primary source: [arXiv](https://arxiv.org/abs/2308.08155). It underpins the composition of agents with LLMs and tools; it is not evidence of detection gains on GOOSE. The cited DOI is that of the preprint and must be identified as such.

## Writing guidance

Separate the published evidence from this work's hypothesis. The combination of attack generation, detection and feedback can be presented as a proposal to be evaluated. Gains in robustness, difficulty evolution, generalization and false-positive reduction depend on the ERENO-AI experiments and must not appear as results already obtained.
