# Referências para a introdução do ERENO-AI

Verificação em 17 de setembro de 2026. As afirmações abaixo se limitam aos resumos e metadados das fontes primárias indicadas. Não constituem uma revisão sistemática nem demonstram ineditismo da arquitetura proposta.

## ERENO

Silvio E. Quincozes, Célio Albuquerque, Diego G. Passos e Daniel Mossé. **ERENO: A Framework for Generating Realistic IEC–61850 Intrusion Detection Datasets for Smart Grids**. IEEE Transactions on Dependable and Secure Computing, 21(4), 3851–3865, 2024. **DOI: 10.1109/TDSC.2023.3336857**.

Fonte primária: [registro da IEEE](https://ieeexplore.ieee.org/document/10339874/). O texto indexado da página da editora informa a publicação antecipada em 4 de dezembro de 2023 e a edição definitiva de julho/agosto de 2024; utilizar 2024 na referência com volume e páginas. DOI e ano da edição também constam do [ORCID de Célio Albuquerque](https://orcid.org/0000-0002-7959-6569).

Afirmações sustentadas: o ERENO gera tráfego sintético segundo especificações IEC 61850 e disponibiliza conjuntos de dados com tráfego normal e sete classes de ataque. O artigo identifica a falta de dados representativos e a utilização de bases proprietárias como obstáculos à avaliação e reprodução de resultados de IDS em subestações. Não atribuir ao ERENO original um ciclo multiagente baseado em LLM nem resultados da proposta ERENO-AI.

## Ataques GOOSE

Taha Selim Ustun, Shaik Mullapathi Farooq e S. M. Suhail Hussain. **A Novel Approach for Mitigation of Replay and Masquerade Attacks in Smartgrids Using IEC 61850 Standard**. IEEE Access, 7, 156044–156053, 2019. **DOI: 10.1109/ACCESS.2019.2948117**.

Fonte primária: [registro institucional com resumo e metadados](https://pure.kfupm.edu.sa/en/publications/a-novel-approach-for-mitigation-of-replay-and-masquerade-attacks-/). Identificador IEEE: 8873588. [DOI](https://doi.org/10.1109/ACCESS.2019.2948117).

Afirmações sustentadas: mensagens GOOSE podem ser objeto de replay e masquerade; a alteração do conteúdo dificulta a distinção entre mensagens legítimas e falsificadas. O artigo analisa esses ataques e implementa uma solução de autenticação e integridade, avaliada em laboratório. Evitar afirmar que toda implementação atual do IEC 61850 é desprotegida ou que o estudo prova consequências físicas em qualquer instalação.

## Detecção em subestações

Silvio E. Quincozes, Célio Albuquerque, Diego Passos e Daniel Mossé. **A survey on intrusion detection and prevention systems in digital substations**. Computer Networks, 184, 107679, 2021. **DOI: 10.1016/j.comnet.2020.107679**.

Fontes primárias: [editora Elsevier](https://www.sciencedirect.com/science/article/pii/S1389128620312895), [cópia na página institucional do autor](https://www.facom.ufu.br/~sequincozes/producoes/quincozes2021survey.pdf).

Afirmações sustentadas: a comunicação digital entre dispositivos de subestações permite monitoramento e controle; os requisitos temporais e as características operacionais tornam inadequada a transposição automática de IDS convencionais. O estudo organiza aspectos de projeto e avaliação de IDS e identifica limitações diante de masquerade. A caracterização do estado da arte é de 2021; não apresentar sua avaliação como levantamento atualizado até 2026.

## Notas complementares verificadas pelo agente principal

B. Biggio, G. Fumera e F. Roli. **Security Evaluation of Pattern Classifiers under Attack**. IEEE Transactions on Knowledge and Data Engineering, 26(4), 984–996, 2014. **DOI: 10.1109/TKDE.2013.57**. Fonte primária: [IEEE](https://ieeexplore.ieee.org/document/6494573). Fundamenta a avaliação de classificadores sob manipulação intencional e condições adversariais. Não prova a eficácia específica do ERENO-AI.

Qingyun Wu et al. **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation**. Preprint arXiv, 2023. **DOI: 10.48550/arXiv.2308.08155**. Fonte primária: [arXiv](https://arxiv.org/abs/2308.08155). Fundamenta a composição de agentes com LLMs e ferramentas; não constitui evidência de ganho de detecção em GOOSE. O DOI citado é do preprint e deve ser identificado como tal.

## Orientação de redação

Separar as evidências publicadas da hipótese deste trabalho. A combinação de geração de ataques, detecção e realimentação pode ser apresentada como proposta a avaliar. Ganhos de robustez, evolução da dificuldade, generalização e redução de falsos positivos dependem dos experimentos do ERENO-AI e não devem aparecer como resultados já obtidos.
