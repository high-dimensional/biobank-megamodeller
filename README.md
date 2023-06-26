# The UCL High-Dimensional Neurology X Data & Insight team university strategics interactive dashboard

**Please note this software is in beta, will likely undergo significant further refinement, and may even contain bugs.**

Software © Dr James K Ruffle | j.ruffle@ucl.ac.uk | High-Dimensional Neurology, UCL Queen Square Institute of Neurology


![Overview](assets/success_model_PTF_black_enlarged_font.svg)
**Image key:** *Node and font size* - grant funding successfully awarded (in £m) | *Edge width and colour* - collaborative grant funding successfully awarded (in £m) | *Blue nodes and edges* - community structure yielded from the nested stochastic block model, formulated as a multivariate hypergraph weighted by i) successful application frequency (number of successful collaborative applications), and ii) amount awarded in successful collaborative applications (in £m).


## Table of Contents
- [What is this repository for?](#what-is-this-repository-for)
- [Installation instructions](#installation-instructions)
- [Usage](#usage)
- [Citation](#citation)
- [Funding](#funding)



## What is this repository for?
**Research capability within an institution is a function of the funding that facilitates it.**

This in itself is herein argued to be a function of:
1. **Illuminating successful collaborations** within the centre, criterion on both
	- maximising academic grant money awarded
	
		*and* 
	- maximising the number of successful applications made

	*Together, these increase researcher, institutional, and grant, 'yield'*

2. **Maximising existing successful collaborations**

3. **Identifing probable - but presently non-existent - successful collaborations** via a [generative graph model](https://dx.doi.org/10.1103/PhysRevX.8.041011) of the underlying community structure of the institution.

![diagram](assets/diagram.png)


## Installation instructions
**Plug in and play in three easy steps**
1) Download the following [HTML dashboard](generative_worktribe_dashboard_d3_js.html)
2) Download the [assets folder](assets)
3) Open [HTML dashboard](generative_worktribe_dashboard_d3_js.html) in your web browser. 

*N.B. Step 1) and 2) can be done with github on the command line with:*
```
git clone https://github.com/high-dimensional/university-strategics.git
```

**Important** - ensure that the [HTML dashboard](generative_worktribe_dashboard_d3_js.html) is in the same directory as the [assets folder](assets). *If this is not done, images will *not* render*. Folder structure on your local machine should be as follows:
```
/local/path/to/university-strategics/
├──HTML dashboard/
├──assets/
      ├──*trajectory*.png
      ├──*sbm*.png
      ├──*tsne*.png
      ├── ...
      ├── ...      
      ├── and so on...
```
Some browers are *significantly faster* in using the [HTML dashboard](generative_worktribe_dashboard_d3_js.html) than others. In our experience, [Google Chrome](https://www.google.com/chrome/) is relatively fast.


## Usage
1. This software serves as a **fully interactive dashboard** highlighting both **existing collaboration within an institution**, but also **presently non-existent collabrations likely to be successful** identified via the [generative graph model](https://dx.doi.org/10.1103/PhysRevX.8.041011).

![overview_schematic](assets/overview_schematic.png)

2. **Individual *nodes (departments)* and *edges (collaborations)* can be highlighted.** *Hovering* over a node or edge will show you a brief schematic on individual (node) or collaborative (edge) success over time. *Clicking* a node or edge will give you detailed dropdown information including on:
	- **Number of successful grant applications** over time, and number of applications submitted, compared to all other centres.
	- **Amount of successfully awarded grant funding** over time, and amount applied for, compared to all other centres.
	- **Observed successful collaborations** with community placement in the [generative Bayesian stochastic block model](https://dx.doi.org/10.1103/PhysRevX.4.011047), and **imputed currently non-existent collaborations likely to be successful** from the [reconstruction of edge errors/omissions with the generative mixed-measured blockstate](https://dx.doi.org/10.1103/PhysRevX.8.041011)
	- **Latent space representation of project free-texts** from [GPT3](https://openai.com/blog/gpt-3-apps) --> [tSNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf), and relation to grant outcome and amount applied for.
	- **Detailed breakdown of all successful grant applications for a given node, in order of financial size.** *N.B. clicking the WorkTribe ID will take you to the respective project, but depending on WorkTribe access privileges may or may not show you the project itself.*
	
![example_breakdown](assets/example_breakdown.png)
	
3. **Nodes and edges can be dragged, moved, and even modified for what they show in real-time.** The dropdown menu in the top-right area of the dashboard allows you to:
	- **Change the labels** given to nodes (*default = department name*), or edges (*default = None*). This can be changed to depict various other parameters, such as the number of successful applications, amount of funding awarded, and over various academic years.
	- **Control the aesthetics** of nodes (*default size = in proportion to financial success from grants*) or edges (*default size = in proportion to collaborative financial grant success*). This can be changed to show various other parameters, such as the number of successful applications, amount of funding awarded, and over various academic years.
	- **Export static images** of the results, in various formats.
	- Modify graph drawing parameters (advanced usage - refer to [gravis](https://robert-haas.github.io/gravis-docs/rst/api/d3.html)).
	
![example_modification](assets/example_modification.png)
	
	
## Usage queries
Any questions, bug reports, or feature requests, should be directed to: Dr James K Ruffle | j.ruffle@ucl.ac.uk

Data courtesy of the UCL strategics team & Prof Parashkev Nachev. 
[GPT3](https://openai.com/blog/gpt-3-apps) --> 
[tSNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) embeddings courtesy of Dr Amy Nelson.


## Citation
*Article to follow.*


## Funding
*TBC*

