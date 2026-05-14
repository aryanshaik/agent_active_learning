# Active Learning Agent: Autoresearch Loop

This is an autonomous active learning loop for molecular property prediction on HMS O2. You are an LLM agent (DeepSeek via opencode). The file you are reading is your full directive. Read the reference paper at the bottom — it describes prior experiments on this exact dataset and task.

## Architecture
- **Encoder**: MiniMol (Morgan fingerprints, 2048-bit, radius 2 → 512-dim embedding).
- **Model**: FFN ensemble (3 members, 2 hidden layers × 512 dim, batch norm, dropout 0.1).
- **Agent**: opencode + DeepSeek API.
- **Compute**: O2 interactive CPU node (8-10 cores, 64 GB RAM).

## Objectives
Maximize the **AUROC on a held-out test set** through iterative active learning. Track AUPRC, hit rate, novelty, and diversity as secondary indicators.

## Problem Setup
- **Total Pool**: ~96,000 unlabeled molecules.
- **AL Budget**: 10 iterations. Selection: 1,000 molecules per iteration.
- **Initial Train**: ~10,000 labeled molecules (10% of pool).
- **Evaluation**: Fixed, held-out test_df.csv (107K molecules).

## Setup (already done)
- Branch: al/may13_v2
- results.tsv: initialized with baseline row
- Data: train_df.csv (10%), pool_df.csv (90%), test_df.csv — all disjoint
- Environment: al-agent conda env activated

## Cost Tracking
After each iteration, add token usage and estimated cost to the results.tsv description field. Track:
- Approximate tokens used (prompt + response) for that iteration's decisions
- Estimated DeepSeek API cost (DeepSeek Flash: ~$0.07/M input tokens, ~$0.28/M output tokens)
- Cumulative cost across the campaign

## Experimentation Loop

LOOP:

1. **Read files**: al_optimizer.py (modifiable), code/minimol_ffn.py (read-only), results.tsv (history).
2. **Modify al_optimizer.py**: Refine acquisition weights, selection size, ensemble size, training hyperparameters. Use the paper's findings.
3. **Git Commit**: git commit -am "<strategy description, include cost estimate>"
4. **Run**:
   ```
   python -u al_optimizer.py --iters 1 --train data/train_df.csv --pool data/pool_df.csv --test data/test_df.csv 2>&1 | tee al_run.log
   ```
5. **Extract Metrics**: Parse FINAL_METRICS line from al_run.log:
   ```
   FINAL_METRICS iter=N: test_auroc=X test_auprc=Y hit_rate=Z
   ```
   Also grep novelty, diversity, true_hit_rate, selected_pos from above.
6. **Log Results**: Append row to results.tsv: commit hash, iteration, metrics, status, description (including token usage and cost).
7. **Decision**: Keep all runs (no git reset). If performance drops, analyze and adjust.

## Constraints
- **No Label Leakage**: Pool labels hidden until selection.
- **Read-only model**: code/minimol_ffn.py is fixed. Only modify al_optimizer.py.
- **Never ask to stop**: Run fully autonomously.

---

# Reference Paper

The following is a draft manuscript from the Farhat Lab describing active learning experiments on M. tuberculosis HTS data — the same dataset and task you are working on. Read it carefully and use the experimental design, acquisition strategies, and findings to inform your decisions.

---

## Active Learning-guided narrow-spectrum antibiotics discovery for infectious disease

Lia, Andrew, Alex, Kee-Lee, Yasha, Jim, Peter, Eric, Linden, Maha

### ABSTRACT

Antibiotic discovery is increasingly reliant on high throughput screening (HTS) for identifying whole bacterial cell growth inhibitors—especially for bacteria with poorly understood drug targets or difficult to penetrate cell walls. HTS can be costly, and the choice of compounds to screen is critical to success. The use of machine learning (ML) to guide HTS compound selection offers promise not only for increasing screening hit rate but also for building more generalizable models able to predict inhibitory effects for out-of-distribution molecules. Such models can greatly accelerate antibiotic discovery when resources are limited, effectively driving down the cost needed to generate new leads. The iterative ML and experimental HTS strategy, also known as Active Learning (AL) has been successfully applied for targeted bioactivity screening, but not yet for whole-cell bacterial bioactivity. Further, it is not known if, within AL, prioritizing compound novelty, inhibition or a combination is needed across metrics of success of ML and whole-cell bacterial bioactive HTS. In this work, we design an AL strategy to prioritize a balance of novelty, growth inhibition and other chemical characteristics for a plate-based selection process compatible with HTS. We explore how our strategy benefits the discovery of bioactives using simulations of retrospective Mycobacterium tuberculosis HTS data (n=20,000 of 114,000). We demonstrate the success of the chosen strategy in a real-time HTS for a Borrelia burgdorferi antibiotic discovery campaign resulting in a 10-fold increase in discovery rate (1.0%) relative to non-AL-assisted plate selection (0.1%). Further, we show that AL can be adapted to guide inhibitor discovery that demonstrate narrow-spectrum activity against the target bacteria while sparing commensals. AL demonstrates promise in navigating large chemical spaces for identifying compounds with multiple desirable properties.

### MAIN TEXT

Antibiotic discovery increasingly relies on high throughput in vitro screening (HTS) for identifying novel compounds that inhibit the growth of whole bacterial cells. However, this approach can be costly, making the prioritization of compounds for screening critical.

Active Learning (AL) is a strategy applied to myriad ML-powered tasks wherein samples are iteratively and selectively labeled for supervised model training depending upon their potential value to the model. AL is particularly apt for HTS due to its iterative format compatible with batched HTS timelines and because experimenters are typically choosing from vast chemical libraries consisting of hundreds of thousands of compounds. As such, AL has demonstrated to be effective in steering such selection processes, balancing sample labeling priority between model improvement and discovery of bioactives. No such methods, however, have applied AL-curated training data on whole-cell activity prediction for causative agents of infectious disease. While many assays screen for compound activity on specific intracellular targets, the success rate does not always translate to whole-cell growth inhibition, often due to the inability to permeabilize the cell wall or account for compensatory cellular effects, rendering whole cell activity hit discovery elusive. Further, we demonstrate our AL-guided models can discover potent B. burgdorferi inhibitors with underrepresented chemical scaffolds.

### RESULTS AND DISCUSSION

**Mtb active learning simulations**

The optimal AL strategy may vary depending on several factors such as sample type, available samples to label, total labeling budget, model architecture, and the model's objective. In our case, as we apply AL to HTS for antibiotic discovery, we are interested in uncovering inhibitors against whole bacterial cells, while also achieving sufficient training sample diversity such that our resultant model can accurately predict inhibitors with underrepresented chemical scaffolds. Here, we explore how the performance of an evidential DMPNN classification model trained with the objective of M. tuberculosis growth inhibition prediction is influenced by AL prioritizing of diversity, novelty, predicted inhibition, and uncertainty. We are operating within a "library" of 114,933 compounds with a target "labeling budget" of ~50%. Thus, we also address how limitations of the candidate library and labeling budget impact common AL priorities.

Using a dataset of 114,933 compounds screened against M. tuberculosis for relative growth inhibition, we tested two acquisition functions relative to random selection–one that prioritizes molecules with the highest probability of growth inhibition ("inhibition"), and one that prioritizes a combination of intra-batch diversity, chemical uniqueness from past iterations, as well as growth inhibition probability ("Diversity-Novelty-Inhibition"). We pre-define batches to simulate a realistic constraint of vendor-manufactured 384-well plate formats common in high-throughput compound screening. We simulated AL iterations to ~50% percent of our total dataset (n=55,040).

Our simulations demonstrate that both Diversity-Inhibition and Inhibition only schemes perform equally better at hit discovery relative to random (Figure 2A, 2B), yet the Diversity-Inhibition scheme outperforms Inhibition-only with a consistently more favorable precision-recall trade-off in models trained with more than ~40,000 compounds (Figure 2C). To highlight the relative gain in generalizability of the Diversity-Inhibition scheme, we separated the test set into 10 bins of ascending "similarity" to the train set (using the maximum cosine similarity of each test molecule relative to each train molecule). Then, the AUPRC is computed for each bin for each respective model. Finally, we compute the area under the AUPRC vs. bin curve, which we call the "Generalizability Score."

Inhibition-only, yet the Diversity-Inhibition-Novelty model demonstrates superior precision and recall on both the entire dataset as well as bioactive-relevant categories all while selecting a more chemically diverse set of bioactive molecules for the benefit of antibiotic discovery efforts.

Many report the value of incorporating model uncertainty as a strategy for AL-based training sample labeling. To explore how this factor interplays with our setup, we incorporated two uncertainty terms (Eqns. 1-3) into the Diversity-Novelty-Inhibition AL strategy. We call this strategy Diversity-Novelty-Inhibition-Uncertainty. When compared with the other two AL strategies on the metrics of percent of hits selected, percent hit rate, and AUPRC, Diversity-Novelty-Inhibition-Uncertainty performs similarly to the other conditions relative to random (Supplementary Figure S1A-C). However, when compared on the grounds of generalizability, Diversity-Novelty-Inhibition-Uncertainty performs similarly to random (Supplementary Figure S1D), perhaps because of our contrived labeling budget of 50,000 precluding our ability to observe potential performance gains reported in other studies. Although the percent of hits selected using Diversity-Novelty-Inhibition-Uncertainty relative to the other strategies is comparable, the novelty of the hits at early AL iterations is notably higher than that for the Diversity-Novelty-Inhibition strategy (Supplementary Figure S1F), which is expected since the uncertainty component is essentially an additional term in the selection scheme that captures novelty–i.e. through the prioritization of compounds for which the model lacks relevant evidence for inhibition prediction.

Finally, we address the library composition and size as a confounding factor to the AL process. To explore this, we plotted "novelty relative to remaining library", calculated as one minus the median cosine similarity between MiniMol fingerprints of each training sample relative to the remaining library molecules, for each condition. We see there comes a point during library selection wherein the contribution from novelty is rapidly exhausted by the Diversity-Inhibition-Novelty scheme (Supplementary Figure S1D). At this point during AL selection, once there is little difference in novelty of the remaining molecules, the rankings lack discriminatory power and the selection essentially becomes random.

Figure 2. Benchmarking of three AL strategies on Mtb HTS data. (A) Percent of total hits screened as a function of number of compounds (B) Number of active compounds screened per number of molecules screened ("hit rate"). (C) AUPRC as a function of number compounds. Each datapoint represents a new training batch selected on the basis of the Random (gray), Diversity-Inhibition-Novelty (blue), or Inhibition (red) acquisition functions. (D) Generalizability score as a function of number compounds for each condition. Holdout data (n=10,010) are binned into 10 bins incrementally more chemically divergent from the train set. The AUPRC of the three most chemically-divergent bins is calculated. The area under the AUPRC vs. bin curve is then calculated for each iteration and is termed "Generalizability Score". (E) Novelty (one minus the min. cosine similarity between MiniMol fingerprint of new hit and each hit from previous iteration) of active compounds screened at each iteration relative to the previous iteration. (F) Fisher's exact test was used to test for significant enrichment of medicinally-relevant chemical moieties in M. tuberculosis inhibitors (n=6,572). Chemical moieties with a p-value < 0.001 and an odds ratio > 1.5 are labeled and color-coded in green. (G) Performance on holdout molecules summarized by F1 score split by membership in 50 medicinally-relevant chemical moieties for the Inhibition-only vs Diversity-Inhibition-Novelty models are plotted. Chemical moieties enriched in hits from our dataset are color-coded in green.

**Real-time B. burgdorferi active learning**

Next, we apply our acquisition function to a parallel E. coli-B. burgdorferi-active compound screening. The gram-negative-defining properties of these bacteria that differ from the gram-positive leaning Mtb render a relatively more durable cell wall less amenable to antibiotics. This presents in HTS experiments as low "hit" rates of 0.1% and 0.5% for E. coli and B. burgdorferi, respectively. This characteristic widens the already gaping class balance discrepancy in inhibition modeling, posing an additional challenge for machine learning applications in HTS. Thus, not only does prioritizing inhibitory molecules accelerate bioactive discovery, but also contributes to improving inhibition modeling.

To begin implementing our AL campaign for narrow spectrum hit discovery, we trained a DMPNN with a Dirichlet head with libraries containing compounds curated for diversity, known bioactivity, and bioavailability (Biomol4, ChemBridge2020, Selleck FDA-Approved 2023, ChemDiv7) (n=10,473). We then used our scoring scheme to select plates from a pool of 296,454 compounds grouped into 906 plates (ICCB-Longwood Screening Facility at Harvard Medical School). For each subsequent training batch, plates within the top five percent of the total plates were selected in batch sizes ranging from 1,760 to 3,500. From batches five through eleven, a set of plates were either selected for screening at random or in a hypothesis-driven manner in batch sizes ranging from 1,760 to 6,000 (Figure 3C). These screening results were incorporated into the model alongside the AL-selected plates. We observe a five fold increase in hit rate of plates selected by the AL format in comparison to the non-AL guided selection (Figure 3 A-C).

Next, we performed enrichment analyses of chemical moieties with known medicinal chemistry relevance to interpret potential patterns learned by the model.

Figure 3. Active Learning-guided sample selection across iterations. Percent Hit Rate (A), cumulative hits (B), and cumulative compounds (C) per model training batch are compared between AL- vs. Scientist-guided compound selection, demonstrating a five-fold improvement in hit discovery rate. (D) Fisher's exact test used to test for significant enrichment of medicinally-relevant chemical moieties in B. burgdorferi inhibitors. Chemical moieties with a p-value < 0.001 and an odds ratio > 1.5 are labeled. (E) Fisher's exact test for enrichment in Scientist-selected plates and (F) AL-selected plates. Moieties enriched within the hits are colored blue. (G) ROCs for AL-only, AL with hit control, and Scientist-only curated train sets are plotted with AUROC values of 0.86, 0.77, and 0.40, respectively. (H) PRCs similarly plotted with AUPRC values of 0.18, 0.10, and 0.03, respectively.

**Virtual Screening and Validation for Narrow-Spectrum Candidates**

Figure 4A: Venn diagrams of proteins across staph, e. coli, lyme. UMAP of MPNN embeddings of ICCB + enamine/medchem express molecules. KeeLee validation of enamine/medchem express hits. KeeLee vs AL only model ability to have predicted validated hits.

### METHODOLOGY

**High throughput screening of M. tuberculosis**

mc2-7000 strain of M. tuberculosis (MTB) starter culture was grown in 7H9 media supplemented with OADC, 0.04% Tyloxapol and 25 ug/ml of pantothenate to the OD600 of 1. For the screening, starter culture cells were diluted into testing media to the OD600 of 0.01 and dispensed into 384 well plates. The first set of 50,000 molecules was tested at a final concentration of 10 uM using M9 minimal media supplemented with 0.5% dextrose, 0.2% sodium acetate, 0.01% NaCl, 0.04% Tyloxapol and 25 ug/ml of pantothenic acid. After 5 days of incubation at 37C, resazurin was added to a final concentration of 0.005 mg/ml to provide viability readout. Plates were incubated for additional 1-2 days at 37C, until negative control wells (1% DMSO) were uniformly converted to resorufin pink. Resorufin fluorescence was recorded by OMEGA Polar Star plate reader with an excitation wavelength of 540 nm, and emission at 590 nm. Percent inhibition for each compound was calculated using positive (1 uM rifampicin) and negative (1% DMSO) controls on each corresponding plate. The second set was tested at a final concentration of 20 uM in the 7H9 media supplemented with 0.5% dextrose, 0.1% sodium acetate pH 7.0, 0.05% tyloxapol, 25 ug/mL pantothenic acid, and 25 ug/mL kanamycin, using both resazurin and luminescence readouts. For the later, MTB mc2-7000 strain was transformed with luxCDABE operon containing plasmid enabling constituent expression of both light producing enzyme and the substrate for it. Resazurin conversion to resofurin was recorded as described above, luminescence was measured by OMEGA Clariostar plate reader at 37C after 5 days of incubation with the compound. Percent inhibition was calculated by normalizing between positive and negative controls included on each plate. Finally, a set of 100,000 molecules, partially overlapping with the first set, was tested under the same growth conditions as set two, using luciferase expressing mc2-7000 strain.

**Model**

We trained a Directed Message Passing Neural Network (DMPNN) model coupled to a feed-forward network and an evidential Dirichlet head within Chemprop 1.7.0. This model architecture was selected based upon our previous benchmark of DMPNN with evidential Dirichlet loss against several state-of-the-art architectures for Mtb inhibition classification tasks. We utilized additional learned features alongside the MPNN representations for the final FFN predictions. The DMPNN has three layers, and the hidden dimension is 300. A dirichlet loss function was used with an evidential regularization of 0.2. The input data are binarized percent growth inhibition data, requiring at least 80% growth inhibition to be considered active. We applied class balancing within ChemProp to account for the low hit rate. After DMPNN training, the embeddings were concatenated to RDkit fingerprints (Batches 1-7) or MiniMol fingerprints (Batches 7 and above). The resulting representations were passed through 2 feed-forward layers using the ReLU activation function. During training, we implemented Chemprop's one-cycle learning rate schedule (initial rate of 0.0001, maximum rate of 0.001) across 200 epochs. The scaffold split strategy was used to partition train, test, and validation sets (80/10/10) split. The final model is an ensemble of the 5 replicates.

**Batch selection: 385-well plates**

Plates are ranked according to the following metrics:
1. Novelty score of compounds with inhibition probability < 0.7
2. Number of novel hits (a molecule within a plate is considered novel if it has a tanimoto similarity of at most 0.5 relative to each molecule in the previous train set)
3. Number of compounds with inhibition probability > 0.7
4. Number of unique "rationales"
5. Number of Butina clusters

Each plate in the compound library receives a ranking for each metric. These ranks are aggregated into a summary rank according to Eqn. 1: Plate Rank = sum(w_j * r_ij) where w_j = {1, 2, 1, 1, 1} and r_ij is the descending rank of molecule i for metric j.

**RETROSPECTIVE ACTIVE LEARNING**

Data: The Mtb growth inhibition dataset comprised 114,933 distinct molecules, of which 6,572 (5.7%) were annotated as M.tb growth inhibitors (compounds or fragments), and 108,361 (94.3%) were annotated as non-inhibitors. We split the data using the k-means split strategy from Chemprop (v2.2.2) data module to stratify compounds eligible for training (n=104,923) vs. used as a test set (n=10,010). Compounds from the Mtb dataset were randomly assigned to groups of 320 to emulate the 384-well HTS plate format. For each simulation, the same 2 plates (n=640 molecules) were used to train the initial model which was used for subsequent AL-based selections.

Model: For our retrospective analysis we utilized the same architecture (DMPNN + evidential Dirichlet head) as that used within the real-time Bb AL workflow except there were no additional molecular features appended to the DMPNN embeddings. For each AL train iteration, the model is trained across 75 epochs and the final model is an ensemble of four replicates.

**Acquisition functions:**

Diversity-Inhibition-Novelty:
1. Novelty score of compounds with inhibition probability < 0.7
2. Number of novel hits (tanimoto similarity <= 0.5 relative to each molecule in previous train set)
3. Number of compounds with inhibition probability > 0.7
4. Number of Butina clusters
w_j = {1, 2, 1, 1}

Inhibition:
1. Average inhibition probability
w_j = {1}

Diversity-Inhibition-Novelty-Uncertainty:
1. Novelty score of compounds with inhibition probability < 0.7
2. Number of novel hits (tanimoto similarity <= 0.5 relative to each molecule in previous train set)
3. Number of compounds with inhibition probability > 0.7
4. Number of Butina clusters
5. Aleatoric uncertainty
6. Epistemic uncertainty
w_j = {1, 2, 1, 1, 1, 1}

Random: Virtual plates were randomly selected from the pool of plates not yet used to train the model.

**VIRTUAL SCREENING**

E. coli and S. aureus data/model: We trained a DMPNN using the same parameters as described for the B. burgdorferi inhibition model for both E. coli and S. aureus inhibition models. For the E. coli model training set, we used the AL batches concurrently tested against E. coli and B. burgdorferi (n=52,913) in addition to public datasets, rendering a train set of 154,403 with 475 hits. For the S. aureus model, we used a public dataset of 327,979 with 1,062 hits.

Ranking Strategy: 1,203,388 compounds were virtually screened for inhibition against B. burgdorferi, E. coli, and S. aureus. These compounds were filtered to only include compounds with a probability of B. burgdorferi inhibition > 0.7, and probability of E. coli and S. aureus inhibition < 0.5. Then, we predicted ADMET properties on the subsetted list. Each compound was ranked on: (1) B. burgdorferi inhibition (ascending), (2) E. coli inhibition (descending), (3) S. aureus inhibition (descending), (4) Aggregate Toxicity Score (descending), (5) Novelty to training data (ascending), (6) Novelty to existing antibiotics (ascending). The Aggregate Toxicity Score was calculated by summing the rank of individual toxicity metrics predicted by ADMET-AI and re-ranking. Novelty relative to training data was calculated by computing one minus the maximum cosine similarity between the MiniMol-derived representation of a given virtually-screened compound and that for the 52,913 training compounds. Novelty relative to existing antibiotics was computed the same way between each virtually-screened compound and entities from ChEMBL of known antibiotic activity. Finally, each of the ranks were summed into an aggregate score which was then re-ranked.
