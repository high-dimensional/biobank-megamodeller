# [The legibility of the human brain](URL): article codebase

Software © Dr James K Ruffle | j.ruffle@ucl.ac.uk | High-Dimensional Neurology, UCL Queen Square Institute of Neurology


## Table of Contents
- [Headline numbers](#headline-numbers)
- [What is this repository for?](#what-is-this-repository-for)
- [Usage](#usage)
- [Citation](#citation)
- [Funding](#funding)


## Headline numbers
- Models trained, validated, and tested across **23810 unique UK Biobank participants**.
- **700 individual models** trained, offering flexibility in data availability spanning volumetric T1, FLAIR, DWI, resting functional connectivity, and non-imaging data (constitutional, psychological, serological, and disease).
- **4526 P100 GPU hours**
- **State-of-the-art sex classification: balanced accuracy on held-out test set 99.7%** (*using T1+FLAIR+DWI neuroimaging*)
- **State-of-the-art age prediction: mean absolute error on held-out test set 2.048%** (*using T1+FLAIR neuroimaging*)

![performance_grid](assets/performance_grid.jpg)
**Model performances.** A) Test set performance for all models across the constitutional (C) – orange, disease (D) – blue, psychology (P) – green, and serology (S) – pink feature domains. Index of performance is given as balanced accuracy for classification targets and R<sup>2</sup> for regression fits. The x-axis of all heatmaps depicts the model target, and y-axis depicts the range of feature inputs. White boxes demarcate the best set of inputs to achieve the greatest out-of-sample model performance.


## What is this repository for?
This is the codebase for the article: [The legibility of the human brain](
Harmonising *large scale multichannel neuroimaging data*, *high-performance hardware*, and a custom-built *general purpose deep learning training pipeline*, we quantify the **individual-level legibility of common biological and pathological characteristics from models of the structurally and functionally imaged human brain**. 

**The process illuminates what can — and what plausibly cannot be — predicted from current neuroimaging with large-scale data and state-of-the-art deep learning architectures.**


![workflow](assets/workflow.jpg)
**Workflow.** A) Data selection and partitioning. B) Mean T1-weighted, FLAIR, DWI images, and rsfMRI connectivity matrix across the full cohort of 23810 participants. C) Layered, nested, generative stochastic block model of modelling targets, with edges depicting the strength of interconnection. Red edges denote positive correlation coefficient (r) values, and blue for those negative. Wider edges denote a greater value of the absolute correlation coefficient, |r|. D) Algorithmic approach for exploring the model target-feature space to distinguish targets that can be reliably predicted from those that cannot, across all possible data inputs. Shown here is also a schematic of the possible data to train with, ranging from non-imaging data across the constitutional (C) – orange, disease (D) – blue, psychology (P) – green, and serology (S) – pink feature domains; and T1/FLAIR volumetric structural imaging; DWI volumetric imaging; and rsfMRI connectivity. These data are passed to individual trainable model blocks: a separate multilayer perceptron (MLP) for both non-imaging data, and rsfMRI connectivity, and a 3D convolutional neural network (CNN) for T1/FLAIR and/or DWI. Model block dense layers are then concatenated and passed to a final MLP for output prediction.


![data_utility](assets/data_utility.jpg)
**Domain-specific effects.** Linear mixed-effects models for predicting out of sample performance (balanced accuracy or R2, where applicable) from structural imaging, functional imaging, and non-imaging domain feature sets. Shown are coefficient plots for models whose targets are A) constitutional, B) psychology, C) disease, and D) serology. Inputs with coefficients whose values are positive are associated with increase model performance (advantageous), whilst features with negative coefficients are associated with weaker performance (detrimental). Asterisks stipulate statistical significance as per standard convention: * denotes p<0.05; ** denotes p<0.01; *** denotes p<0.001.



## Usage
Open [](Interactive_results.html)

Code for all model training and post-hoc analysis is open sourced [here](code/)
	
![html_tutorial](assets/html_tutorial.jpg)
**Visual network analysis plots of feature relationships and model performances.** A) Graph of target features, with nodes sized by the absolute correlation coefficient (|r|)-weighted eigenvector centrality (EC), and edges sized according to |r|. B) Graph of target features, with nodes sized by the maximum information coefficient (MIC)-weighted eigenvector centrality (EC), and edges sized according to the MIC. C) Graph of target features, with nodes sized by the maximum balanced accuracy across all models (BA), with edges sized according to the mean inverse Euclidean distance of all input combinations between each pair of targets. For all panels we depict the top 60% of edges for visualisation purposes. Note that all graphs are made available as fully interactive and customizable HTML objects within the supplementary material.


## Usage queries
Via github issue log or email to j.ruffle@ucl.ac.uk


## Citation
If using these works, please cite the following [paper](https://arxiv.org/abs/X):

James K Ruffle, Robert Gray, Samia Mohinta, Guilherme Pombo, Chaitanya Kaul, Harpreet Hyare, Geraint Rees, Parashkev Nachev. The legibility of the human brain. arXiv. 2023. DOI X



## Funding
The Wellcome Trust; UCLH NIHR Biomedical Research Centre; Medical Research Council; Guarantors of Brain; NHS Topol Digital Fellowship.
![funders](assets/funders.png)

