# Suicide and Depression Classification with NLP (university reimplementation of SDCNL)

This repository is a reimplementation of the paper ["Deep Learning for Suicide and Depression Identification with Unsupervised Label Correction"](https://arxiv.org/abs/2102.09427) by Ayaan Haque*, Viraaj Reddi* and Tyler Giallanza from Saratoga High School and Princeton University, published at [ICANN](https://e-nns.org/icann2021/) 2021.

Original project page: https://ayaanzhaque.github.io/SDCNL/
Original code: https://github.com/ayaanzhaque/SDCNL

I (Sunil Boopalan) reimplemented this method as a university project at SRM Institute of Science and Technology in 2021-22, following the original paper and code. The method and dataset are the original authors' work; the code in this repository is my reimplementation.

Correction and apology (October 2026): an earlier version of this README presented the paper and method as mine and replaced the original authors' names with my own. I apologise to Ayaan Haque, Viraaj Reddi and Tyler Giallanza for not crediting their work. As a student in 2021, I wrongly assumed that reimplementing a published method made the work my own. It does not, and this was a careless mistake rather than an attempt to mislead. The work was never submitted or published as my own research. The credit above is now correct.

## About the method

The authors' method, SDCNL, separates depression from suicidal ideation in Reddit posts. Because subreddit labels are noisy, it adds an unsupervised label correction step before classification:

1. Posts are converted into numerical embeddings using pre-trained language models.
2. The embeddings are reduced in dimension and clustered without using the labels.
3. Where the clustering confidently disagrees with the original subreddit label, the label is corrected.
4. A deep learning classifier is trained on the corrected labels.

![Pipeline (from the original authors)](https://github.com/SunilBoopalan/Suicide-Depression-Classification-with-nlp/blob/main/figs/pipeline.png?raw=true)

## Dataset

The dataset, created by the original authors, contains 1,895 Reddit posts from r/SuicideWatch (labelled suicidal) and r/depression (labelled depressed), provided in the data folder with a train/test split. The web-scraper.py script can be used to collect more recent posts through the Reddit API.

## Code

My reimplementation, following the structure of the authors' pipeline:

- word_embeddings.py: converts raw text into embeddings
- clustering-based-label-correction.py: dimensionality reduction and clustering
- threshold-based-correction.py: decides whether to keep the original label or use the clustering label
- classifiers.py: trains the final classifier on the corrected labels

## Results

Results from my reimplementation.

![ROC curves](https://github.com/SunilBoopalan/Suicide-Depression-Classification-with-nlp/blob/main/figs/finalroc.png?raw=true)

![Results table](https://github.com/SunilBoopalan/Suicide-Depression-Classification-with-nlp/blob/main/figs/finaltable.png?raw=true)

## Citation

If you use this work, please cite the original paper:

Ayaan Haque, Viraaj Reddi and Tyler Giallanza. "Deep Learning for Suicide and Depression Identification with Unsupervised Label Correction." International Conference on Artificial Neural Networks (ICANN), 2021. arXiv:2102.09427.
