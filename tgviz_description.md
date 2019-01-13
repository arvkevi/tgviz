### Goal
The goal of this project was to quickly assess whether a small set of genomic markers could roughly differentiate between sample populations in the 1000 Genomes Project data.

### Graph
This graph is a 3D scatter plot representing genotypes from the [1000 Genomes Project](http://www.internationalgenome.org/home) that have been processed by a dimensionality reduction algorithm, with data points colored according to a sample's population.

* A subset of the samples' single nucleotide polymorphism(s), or, SNP(s) have been parsed from the [publicly available `.bcf` files](ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/bcf_files/).  
* The subset of `SNPs` were chosen from two publications which identified **ancestry informative** SNPs (AISNPs).
  * Set of 55 AISNPs. [Progress toward an efficient panel of SNPs for ancestry inference](https://www.ncbi.nlm.nih.gov/pubmed?db=pubmed&cmd=Retrieve&dopt=citation&list_uids=24508742). Kidd et al. 2014
  * Set of 128 AISNPs. [Ancestry informative marker sets for determining continental origin and admixture proportions in common populations in America.](https://www.ncbi.nlm.nih.gov/pubmed?cmd=Retrieve&dopt=citation&list_uids=18683858). Kosoy et al. 2009 (Seldin Lab)
* Samples' genotypes are one-hot encoded and and processed by a dimensionality reduction algorithm. This step reduces the number of dimensions to 3, enabling plotting.

### Parameters
**Set of AISNPs to use**  
* `Kidd 55 AISNPs`: Subset the 1kG data to the 55 SNPs listed in the manuscript.
* `Seldin 128 AISNPs`: Subset the 1kG data to the 128 SNPs listed in the manuscript.

**Dimensionality Reduction Algorithm**
* `PCA`: *Principal Component Analysis*
  * Fastest, old algorithm
  * [scikit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
* `T-SNE`: *t-Distributued Stochastic Neighbor Embedding*
  * Slow
  * [sckit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) -- *not used*
  * [Multicore-TSNE implementation](https://github.com/DmitryUlyanov/Multicore-TSNE) -- *used here*
* `UMAP`: *Uniform Manifold Approximation and Projection*
  * Faster than t-SNE.
  * [umap-learn implementation](https://umap-learn.readthedocs.io/en/latest/)

**Population Resolution**
* `Super Population`: One of {AFR, AMR, EAS, EUR, SAS}.
* `Population`: One of the 26 populations listed [here](http://www.internationalgenome.org/faq/which-populations-are-part-your-study/).

### Interactive
When you click on a data point in the 3D graph, details such as `sample accession`, `population`, `super population`, and `gender` are displayed on the right. The sample-level information is accessible from `integrated_call_samples_v3.20130502.ALL.panel`, found [here](ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/).

### Code  
The code used to process this data is available on [GitHub]().
If interested in replicating this type of analysis, please follow along using the [notebooks]().

Can be used as a package?  
