
[//]: # (Try to put references in harvard style for consistency.)

# aeon projects: ongoing or potential

`aeon` runs a range of short to medium duration projects that involve
developing or using aeon and interacting with the community and the code
base. These projects are designed for internships, usage as part of
undergraduate/postgraduate projects at academic institutions, and as options for
programs such as [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/).

For those interested in undertaking a project outside these scenarios, we recommend
joining the [Slack](https://join.slack.com/t/aeon-toolkit/shared_invite/zt-22vwvut29-HDpCu~7VBUozyfL_8j3dLA)
and discussing with the community. We aim to run schemes to
help new contributors to become more familiar with `aeon`, time series machine learning
research, and open-source software development.

All the projects listed will require knowledge of Python 3 and Git/GitHub. The
majority of them will require some knowledge of machine learning and time series.

## Current aeon projects

This is a list of some of the projects we are interested in running in 2024/25. Feel
free to propose your own project ideas, but please discuss them with us first. We have
an active community of researchers and students who work on `aeon`. Please get in touch
via Slack if you are interested in any of these projects or have any questions.

We will more widely advertise funding opportunities as and when they become available.

Most projects can be extended, possibly into a research project that may lead to
publication. These projects are for anyone, from core devs to those completely new
to open source. We list projects by time series task

[Classification](#classification)
1. Optimizing the Shapelet Transform for classification and similarity search
2. EEG classification with aeon-neuro
3. Implement TS-CHIEF
4. Improved HIVE-COTE implementation.
5. Compare distance based classification.

[Forecasting](#forecasting)
1. Machine Learning for Time Series Forecasting
2. Deep Learning for Time Series Forecasting
3. Implement ETS forecasters in aeon

[Clustering](#clustering)
1. Density peaks clustering algorithm
2. Deep learning based clustering algorithms

[Anomaly Detection](#anomaly-detection)
1. Anomaly detection with the Matrix Profile, MERLIN and MADRID

[Segmentation](#segmentation)
1. Time series segmentation

[Transformation](#transformation)
1. Improve ROCKET family of transformers
2. Implement channel selection algorithms

[Visualisation](#visualisation)
1. Explainable AI with the shapelet transform

[Regression](#regression)
1. Adapt forecasting regressors to time series extrinsic regression.
2. Adapt HIVE-COTE for regression.

[Documentation](#documentation)
1. Improve automated API documentation

### Classification

#### 1. Optimizing the Shapelet Transform for Classification and Similarity Search (listed for GSoC 2024)

Mentors : Antoine Guillaume ({user}`baraline`) and Tony Bagnall ({user}`TonyBagnall`)

##### Related Issues
[#186](https://github.com/aeon-toolkit/aeon/issues/186)
[#324](https://github.com/aeon-toolkit/aeon/issues/324)
[#894](https://github.com/aeon-toolkit/aeon/issues/894)
[#973](https://github.com/aeon-toolkit/aeon/issues/973)
[#1184](https://github.com/aeon-toolkit/aeon/issues/1184)
[#1322](https://github.com/aeon-toolkit/aeon/issues/1322)


##### Description

A shapelet is defined as a time series subsequence representing a pattern of interest
that we wish to search for in time series data. Shapelet-based algorithms can be used
for a wide range of time series tasks. In this project, we will focus on its core
application, which is to create an embedding of the input time series.

Our goal in this project will be to optimize the code related to the shapelet
transform method, which takes as input a set of shapelets and a time series dataset,
and give as output a tabular dataset containing the features characterizing the
presence (or absence) of each shapelet in the input time series (more information
in [1] and [2]).

Similarity search is another field of time series, which has proposed greatly optimized
algorithms (see [3] and [4]) for the task of finding the best matches of a subsequence
inside another time series. As this task is extremely similar to what is done in the
shapelet transform, we want to adapt these algorithms to the context of shapelets,
in order to achieve significant speed-ups.

##### Project stages

To achieve this goal, with the assistance of the mentor, we identify the following
steps for the mentee:

1. Learn about aeon best practices, coding standards and testing policies.
2. Study the shapelet transform algorithm and how it is related to the task of
similarity search.
3. Study the similarity search algorithms for the Euclidean distance and the
computational optimization they use.
4. Propose a candidate implementation for to increase the performance of the
computations made by a single shapelet. This can be made with the help of the existing
implementation of the similarity search module in `aeon`.
5. Measure the performance of this first candidate implementation against the current
approach.
6. Implement this solution to the shapelet transform algorithm, which uses multiple
shapelets.
7. Benchmark the implementation against the original shapelet transform algorithm.
8. If time, generalize this new algorithm to the case of dilated shapelets (see [5]).

##### Expected Outcomes

We expect the mentee to engage with the aeon community and produce a more performant
implementation for the shapelet transform that gets accepted into the toolkit.

##### References

1. Hills, J., Lines, J., Baranauskas, E., Mapp, J. and Bagnall, A., 2014.
Classification of time series by shapelet transformation. Data mining and knowledge
discovery, 28, pp.851-881.
2. Bostrom, A. and Bagnall, A., 2017. Binary shapelet transform for multiclass time
series classification. Transactions on Large-Scale Data-and Knowledge-Centered Systems
XXXII: Special Issue on Big Data Analytics and Knowledge Discovery, pp.24-46.
3. Yeh, C.C.M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H.A., Silva, D.F.,
Mueen, A. and Keogh, E., 2016, December. Matrix profile I: all pairs similarity joins
for time series: a unifying view that includes motifs, discords and shapelets. In 2016
IEEE 16th international conference on data mining (ICDM) (pp. 1317-1322). Ieee.
4. Zhu, Y., Zimmerman, Z., Senobari, N.S., Yeh, C.C.M., Funning, G., Mueen, A., Brisk,
P. and Keogh, E., 2016, December. Matrix profile ii: Exploiting a novel algorithm and
gpus to break the one hundred million barrier for time series motifs and joins. In 2016
IEEE 16th international conference on data mining (ICDM) (pp. 739-748). IEEE.
5. Guillaume, A., Vrain, C. and Elloumi, W., 2022, June. Random dilated shapelet
transform: A new approach for time series shapelets. In International Conference on
Pattern Recognition and Artificial Intelligence (pp. 653-664). Cham: Springer
International Publishing.

#### 2. EEG classification with aeon-neuro

Mentors: Tony Bagnall ({user}`TonyBagnall`) and Aiden Rushbrooke

##### Related Issues
[#18](https://github.com/aeon-toolkit/aeon-neuro/issues/18)
[#19](https://github.com/aeon-toolkit/aeon-neuro/issues/19)
[#24](https://github.com/aeon-toolkit/aeon-neuro/issues/24)



##### Description

EEG (Electroencephalogram) data are high dimensional time series that are used in
medical, psychology and brain computer interface research. For example, EEG are
used to detect epilepsy and to control devices such as mice. There is a huge body
of work on analysing and learning from EEG, but there is a wide disparity of
tools, practices and systems used. This project will help members of the `aeon`
team who are currently researching techniques for EEG classification [1] and
developing an aeon sister toolkit, [``aeon-neuro``](https://github.com/aeon-toolkit/aeon-neuro). We will work together to
improve the structure and documentation for aeon-neuro, help integrate the
toolkit with existing EEG toolkits such as MNE [2], provide interfaces to standard data
formats such as BIDS [3] and help develop and assess a range of EEG classification
algorithms.

##### Project stages

1. Learn about aeon best practices, coding standards and testing policies.
2. Study the existing techniques for EEG classification.
3. Implement or wrap standard EEG processing algorithms.
4. Evaluate aeon classifiers for EEG problems.
5. Implement alternatives transformations for preprocessing EEG data.
6. Help write up results for a technical report/academic paper (depending on outcomes).

##### Expected Outcomes

We would expect a better documented and more integrated aeon-neuro toolkit with
better functionality and a wider appeal.

##### References

1. Aiden Rushbrooke, Jordan Tsigarides, Saber Sami, Anthony Bagnall,
Time Series Classification of Electroencephalography Data, IWANN 2023.
2. MNE Toolkit, https://mne.tools/stable/index.html
3. The Brain Imaging Data Structure (BIDS) standard, https://bids.neuroimaging.io/

#### 3. Improved HIVE-COTE implementation

Mentors: Matthew Middlehurst ({user}`MatthewMiddlehurst`) and Tony Bagnall
({user}`TonyBagnall`)

##### Related Issues
[#663](https://github.com/aeon-toolkit/aeon/issues/663)

##### Description

The Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) [1,2,3]
is a time series classifier that has claims to be state-of-the-art,
particularly in terms of probabilistic estimates [4]. There have been several iterations
that use different base classifiers but they all share the same design basic:
classifiers using different representations are combined in a weighted meta-ensemble
[4]. There are two HIVE-COTE implementations currently in `aeon`: HIVE-COTEV1 [2]
and HIVECOTEV2 [3]. This project will involve combining these into a single
estimator, modularising the ensemble stage and possibly experimenting with
alternative structures. This can easily develop into a research project.

##### Project stages

1. Learn about `aeon` best practices, coding standards and testing policies.
2. Study the HIVE-COTE algorithm and previous `aeon` implementation.
3. Combine the HIVE-COTEV1 and HIVE-COTEV2 classifiers into a single classifier
   configurable into different versions through the constructor.
4. Restructure the ensemble stage to allow easy experimentation with variants.

##### References

1. A Bagnall, J Lines, J Hills, A Bostrom, Time-series classification with COTE: the
   collective of transformation-based ensembles, IEEE Transactions on Knowledge and
   Data Engineering 27 (9), 2522-2535
2. M Middlehurst, J Large, M Flynn, J Lines, A Bostrom, A Bagnall, HIVE-COTE 2.0: a
   new meta ensemble for time series classification, Machine Learning 110 (11),
   3211-3243
3. J Lines, S Taylor, A Bagnall, Time series classification with HIVE-COTE: The
   hierarchical vote collective of transformation-based ensembles, ACM Transactions
   on Knowledge Discovery from Data (TKDD) 12 (5), 1-35
4. Middlehurst, M., Schäfer, P. and Bagnall, A., 2023. Bake off redux: a review and
experimental evaluation of recent time series classification algorithms. arXiv preprint
arXiv:2304.13029.

#### 4. Compare distance based classification and regression

Mentors: Chris Holder ({user}`cholder`) and Tony Bagnall
({user}`TonyBagnall`)

##### Related Issues
[#423](https://github.com/aeon-toolkit/aeon/issues/423)
[#424](https://github.com/aeon-toolkit/aeon/issues/424)
[#425](https://github.com/aeon-toolkit/aeon/issues/425)
[#426](https://github.com/aeon-toolkit/aeon/issues/426)
[#427](https://github.com/aeon-toolkit/aeon/issues/427)
[#488](https://github.com/aeon-toolkit/aeon/issues/488)

##### Description
Distance based algorithms are popular for time series classification and regression.
However, the evaluation of distance functions for classification have not
comprehensively covered all possible uses. For example, there has not been a proper
bake off for using elastic distance with support vector machines or with tuning
distance functions and classifiers in combination. This project will combine
implementing alternative distance functions and comparing performance on the UCR
datasets.


### Forecasting

#### 1. Machine Learning for Time Series Forecasting

Mentors: Tony Bagnall ({user}`TonyBagnall`) and Leo Tsaprounis ({user}`ltsaprounis`) .

##### Related Issues
[#265](https://github.com/aeon-toolkit/aeon/issues/265)


##### Description

This project will investigate algorithms for forecasting based on traditional machine
learning (tree based) and time series machine learning (transformation based). Note
this project will not involve deep learning based forecasting. It will involve
helping develop the `aeon` framework to work more transparently with ML algorithms,
evaluating regression algorithms already in `aeon`[1] for forecasting problems and
implementing at least one algorithm from the literature not already in aeon, such as
SETAR-Tree [3].

##### Project Stages
1. Learn about aeon best practices, coding standards and testing policies.
2. Adapt the M competition set up [2] for ML experimental framework to assess time
   series regression algorithms [1].
3. Implement a machine learning forecasting algorithm [3]

##### Expected Outcomes

1. Contributions to the new experimental aeon forecasting module.
2. Implementation of a machine learning forecasting algorithms.
3. Help write up results for a technical report/academic paper (depending on outcomes).

##### Skills Required

1. Python 3
2. Git and GitHub
3. Some machine learning and/or forecasting background (e.g. taught courses or
   practical experience)

##### References

1. Guijo-Rubio, D.,Middlehurst, M., Arcencio, G., Furtado, D. and Bagnall, A.
Unsupervised Feature Based Algorithms for Time Series Extrinsic Regression,
arXiv2305.01429, 2023
2. https://forecasters.org/resources/time-series-data/
3. Godahewa, R., Webb, G.I., Schmidt, D. et al. SETAR-Tree: a novel and accurate
tree algorithm for global time series forecasting. Mach Learn 112, 2555–2591 (2023).
https://link.springer.com/article/10.1007/s10994-023-06316-x

#### 2. Deep Learning for Time Series Forecasting

Mentors: Tony Bagnall ({user}`TonyBagnall`)  and Ali Ismail-Fawaz ({user}
`hadifawaz1999`)

##### Description

Deep learning has become incredibly popular for forecasting, see [1] for an
introduction. This project will involve taking one or more recently proposed
algorithms, implementing them in aeon, then performing an extensive experimental
comparison against traditional and machine learning algorithms. As part of this, we
will collate results from the M Competitions [2]

##### Project Stages
1. Learn about aeon best practices, coding standards and testing policies.
2. Adapt the M competition set up [2] for deep learning.
3. Implement a deep learning forecasting algorithm after discussion with mentors.

##### Expected Outcomes

1. Collated M competition results and partial reproduction.
2. Extend the forecasting module to include at least one deep forecaster.

##### References

1. [ECML 2024 Tutorial](https://lovvge.github.io/Forecasting-Tutorial-ECML-2023/)
2. [M Competitions](https://forecasters.org/resources/time-series-data/)


#### 3. Implement ETS forecasters

Mentors: Tony Bagnall ({user}`TonyBagnall`)  and Leo Tsaprounis ({user}`ltsaprounis`)
Exponential smoothing (ETS) is a popular family of algorithms for forecasting, and
the ETS framework by Hyndman et al. [1] covers 30 possible models for time series
with different types of Error, Trend, and Seasonal components.
we already have an (Auto)ETS model in aeon, but it’s wrapping statsmodels. We would
like our own bespoke, optimised implementation based on the R implementation.

##### Project Stages
1. Learn about aeon best practices, coding standards and testing policies.
2. Survey and benchmark existing implementations of ETS forecasting.
3. Implement basic implementations optimised for numba.
4. Extended implementation to include modern refinements.


##### References

1. Hydman et al. [Forecasting with Exponential Smoothing The State Space Approach](https://link.springer.com/book/10.1007/978-3-540-71918-2)
2. [Smooth R Package](https://github.com/config-i1/smooth)
3. Svetunkov, [Forecasting and Analytics with the Augmented Dynamic Adaptive Model
   (ADAM)](https://openforecast.org/adam/)

### Clustering

#### 1. Density peaks clustering algorithm

Mentors: Tony Bagnall ({user}`TonyBagnall`) and Chris  Holder (`@chrisholder`).

##### Description

The clustering module in `aeon`, up until now, clusters using time series
specific distance functions with partitional clustering algorithms such as k-means and
k-medoids. An alternative clustering algorithm is density peaks (DP) [1]. This
clustering
algorithm has the benefit of not having to label all cases as cluster members, which
means it can easily be adapted to anomaly detection [2]. It is
a general purpose clustering algorithm that is not available in scikit learn. This
project will implement the algorithm based on Java and matlab implementations then
compare performance against partitional clustering for time series clustering.

##### Project Stages

1. Research and understand how DP works.
2. Implement DP as an aeon estimator.
3. Test the implementation against other implementations for correctness.
4. Compare against alternative TSCL algorithms

##### Expected Outcomes

1. A well documented, tested and efficient implementation of DP
2. Possible extensions to reflect recent research [2] with specific time series
   components [3].
2. Contributions to a comparative study and paper.

##### References
1. Rodriguez, A., & Laio, A. Clustering by Fast Search and Find of
Density Peaks. [Science](https://www.science.org/doi/10.1126/science.1242072), 344
   (6191), 1492-1496, 2014.
2. Chen, L., Gao, S. & Liu, B. An improved density peaks clustering algorithm
based on grid screening and mutual neighborhood degree for network anomaly detection.
Sci Rep 12, 1409 (2022) [DOI](https://doi.org/10.1038/s41598-021-02038-z)
3. Begum et al. A General Framework for Density Based  Time Series Clustering
   Exploiting a Novel Admissible Pruning Strategy, [arXiv](https://arxiv.org/ftp/arxiv/papers/1612/1612.00637.pdf)


#### 2. Deep learning for clustering

Mentors: Tony Bagnall ({user}`TonyBagnall`)  and Ali Ismail-Fawaz ({user}
`hadifawaz1999`)

The clustering module in `aeon`, up until now, primarily consists of distance-based
partitional clustering algorithms. Recently, we introduced a deep clustering module,
incorporating distance-based algorithms in the latent space.

The objective of this project is to enhance `aeon` by incorporating more deep learning
approaches for time series clustering. The specific goal is to implement and assess
InceptionTime [1] and its recent variants as a clustering algorithm, and contribute to
an ongoing collaborative effort into a bake off for clustering. More widely, there
are a broad range of deep learning clustering approaches we could consider [2].

##### Project Stages

1. Research and understand clustering time series and deep learning based approaches.
2. Implement inception time as an aeon clusterer.
3. Compare performance of deep learning clusterers to distance based algorithms.

[1] Fawaz et al. InceptionTime: Finding AlexNet for time series classification
Published: 07 September 2020 Volume 34, pages 1936–1962, (2020)
[2] Deep learning forecasting [tutorial](https://lovvge.github.io/Forecasting-Tutorial-ECML-2023/)

### Anomaly detection


#### 1. Anomaly detection with the Matrix Profile, MERLIN and MADRID

Mentors: Matthew Middlehurst ({user}`MatthewMiddlehurst`)

##### Description

`aeon` is looking to extend its module for time series anomaly detection. The
end goal of this project is to implement the Matrix Profile [1][2] and MERLIN [3]
algorithms, but suitable framework for anomaly detection in `aeon` will need to be
designed first. The mentee will help design the API for the anomaly detection module
and implement the Matrix Profile and MERLIN algorithms.

Usage of external libraries such as `stumpy` [4] is possible for the algorithm
implementations, or the mentee can implement the algorithms from scratch using `numba`.
There is also scope to benchmark the implementations, but as there is no existing
anomaly detection module in `aeon`, this will require some infrastructure to be
developed and is subject to time and interest.

##### Project stages

1. Learn about `aeon` best practices, coding standards and testing policies.
2. Familiarise yourself with similar single series experimental modules in `aeon` such
as segmentation and similarity search.
3. Help design the API for the anomaly detection module.
4. Study and implement the Matrix Profile for anomaly detection and MERLIN algorithms
using the new API.
5. If time allows and there is interest, benchmark the implementations against the
original implementations or other anomaly detection algorithms.

##### Project Outcome

As the anomaly detection is a new module in `aeon`, there is very little existing code
to compare against and little infrastructure to evluate anomaly detection algorithms.
The success of the project will be evaluated by the quality of the code produced and
engagement with the project and the `aeon` community.

##### References

1. Yeh, C.C.M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H.A., Silva, D.F.,
Mueen, A. and Keogh, E., 2016, December. Matrix profile I: all pairs similarity joins
for time series: a unifying view that includes motifs, discords and shapelets. In 2016
IEEE 16th international conference on data mining (ICDM) (pp. 1317-1322). Ieee.
2. Lu, Y., Wu, R., Mueen, A., Zuluaga, M.A. and Keogh, E., 2022, August.
Matrix profile XXIV: scaling time series anomaly detection to trillions of datapoints
and ultra-fast arriving data streams. In Proceedings of the 28th ACM SIGKDD Conference
on Knowledge Discovery and Data Mining (pp. 1173-1182).
3. Nakamura, T., Imamura, M., Mercer, R. and Keogh, E., 2020, November. Merlin:
Parameter-free discovery of arbitrary length anomalies in massive time series archives.
In 2020 IEEE international conference on data mining (ICDM) (pp. 1190-1195). IEEE.
4. Law, S.M., 2019. STUMPY: A powerful and scalable Python library for time series data
mining. Journal of Open Source Software, 4(39), p.1504.

### Segmentation

#### 1. Time series segmentation

Mentors: Tony Bagnall ({user}`TonyBagnall`)

##### Description

The time series segmentation module contains a range of algorithms for segmenting time
series. The goal of this project is to extend the functionality of segmentation in
`aeon` and develop tools for comparing segmentation algorithms.

##### Project stages

1. Learn about `aeon` best practices, coding standards and testing policies.
2. Study the existing segmentation algorithms in `aeon`.
3. Implement existing segmentation algorithms, e.g.
https://github.com/aeon-toolkit/aeon/issues/948
4. Implement tools for comparing segmentation algorithms
5. Conduct a bake off of segmentation algorithms on a range of datasets.

##### Project Outcome

As with all research programming based projects, progress can be hindered by many
unforseen circumstances. Success will be measured by engagement, effort and
willingness to join the community rather than performance of the algorithms.

##### References

1. Allegra, M., Facco, E., Denti, F., Laio, A. and Mira, A., 2020. Data segmentation
based on the local intrinsic dimension. Scientific Reports, 10(1), p.16449.
2. Ermshaus, A., Schäfer, P. and Leser, U., 2023. ClaSP: parameter-free time series
segmentation. Data Mining and Knowledge Discovery, 37(3), pp.1262-1300.
3. Hallac, D., Nystrup, P. and Boyd, S., 2019. Greedy Gaussian segmentation of
multivariate time series. Advances in Data Analysis and Classification, 13(3),
pp.727-751.
4. Matteson, D.S. and James, N.A., 2014. A nonparametric approach for multiple change
point analysis of multivariate data. Journal of the American Statistical Association,
109(505), pp.334-345.
5. Sadri, A., Ren, Y. and Salim, F.D., 2017. Information gain-based metric for
recognizing transitions in human activities. Pervasive and Mobile Computing, 38,
pp.92-109.

### Transformation

#### 1. Improve ROCKET family of transformers

Mentors: Ali Ismail-Fawaz ({user}`hadifawaz1999`) and Matthew Middlehurst
({user}`MatthewMiddlehurst`)
[#208](https://github.com/aeon-toolkit/aeon/issues/208)
[#214](https://github.com/aeon-toolkit/aeon/issues/214)
[#313](https://github.com/aeon-toolkit/aeon/issues/313)
[#1126](https://github.com/aeon-toolkit/aeon/issues/1126)
[#1248](https://github.com/aeon-toolkit/aeon/issues/1248)


##### Description

The ROCKET algorithm [1] is a very fast and accurate transformation designed for time
series classification. It is based on a randomly initialised convolutional kernels that
are applied to the time series and used to extract summary statistics. ROCKET has
applications to time series classification, extrinsic regression and anomaly detection,
but as a fast and unsupervised transformation, it has potential to a wide range of
other time series tasks.

`aeon` has implementations of the ROCKET transformation and its variants, including
MiniROCKET [2] and MultiROCKET [3]. However, these implementations have room for
improvement ([#208](https://github.com/aeon-toolkit/aeon/issues/208)). There is scope
to speed up the implementations, and the amount of varients is likely unnecessary and
could be condensed into higher quality estimators.

This projects involves improving the existing ROCKET implementations in `aeon` or
implementing new ROCKET variants. The project will involve benchmarking to ensure that
the new implementations are as fast and accurate as the original ROCKET algorithm and
potentially to compare to other implementations ([#214](https://github.com/aeon-toolkit/aeon/issues/214)).
Besides improving the existing implementations, there is scope to implement the HYDRA
algorithm [4] or implement GPU compatible versions of the algorithms.

##### Project Stages

1. Learn about `aeon` best practices, coding standards and testing policies.
2. Study the ROCKET, MiniROCKET, MultiROCKET algorithms.
3. Study the existing ROCKET implementations in `aeon`.
4. Merge and tidy the ROCKET implementations, with the aim being to familiarise the
mentee with the `aeon` pull request process.
5. Implement one (or more) of the proposed ROCKET implementation improvements:
   * Significantly alter the current ROCKET implementations with the goal of
   speeding up the implementation on CPU processing.
   * Implement a GPU version of some of the ROCKET transformers, using either
   `tensorflow` or `pytorch`.
   * Extend the existing ROCKET implementations to allow for the use of unequal length
   series.
   * Implement the HYDRA algorithm.
6. Benchmark the implementation against the original ROCKET implementations, looking at
booth speed of the transform and accuracy in a classification setting.

##### Project Outcomes

Success of the project will be assessed by the quality of the code produced and an
evaluation of the transformers in a classification setting. None of the implementations
should significantly degrade the performance of the original ROCKET algorithm in terms
of accuracy and speed. Regardless, effort and engagement with the project and the
`aeon` community are more important factors in evaluating success.

##### References

1. Dempster, A., Petitjean, F. and Webb, G.I., 2020. ROCKET: exceptionally fast and
accurate time series classification using random convolutional kernels.
Data Mining and Knowledge Discovery, 34(5), pp.1454-1495.
2. Dempster, A., Schmidt, D.F. and Webb, G.I., 2021, August. Minirocket: A very fast
(almost) deterministic transform for time series classification. In Proceedings of the
27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 248-257).
3. Tan, C.W., Dempster, A., Bergmeir, C. and Webb, G.I., 2022. MultiRocket: multiple
pooling operators and transformations for fast and effective time series classification.
Data Mining and Knowledge Discovery, 36(5), pp.1623-1646.
4. Dempster, A., Schmidt, D.F. and Webb, G.I., 2023. Hydra: Competing convolutional
kernels for fast and accurate time series classification. Data Mining and Knowledge
Discovery, pp.1-27.

#### 2. Implement channel selection algorithms

Related issues:
[#1270](https://github.com/aeon-toolkit/aeon/issues/1270)
[#1467](https://github.com/aeon-toolkit/aeon/issues/1467)

Channel selection in this context is the process of reducing the number of channels
in a collection of time series for classification, clustering or regression. This
project looks at filter based approaches to speed up multivariate time series
classification (MTSC) of high dimensional series. Standard approaches for
classifying high dimensional data are to
employ a filter to select a subset of attributes or to transform the data into a lower
dimensional feature space using, for example, principal component analysis. Our
focus is on dimensionality reduction through filtering. For MTSC, filtering is
generally accepted to be selecting the most important dimensions to use before
training the classifier. Dimension selection can, on average, either increase, not
change or decrease the accuracy of classification. The first case implies that the
higher dimensionality is confounding the classifier’s discriminatory power. In the
second case it is often still desirable to filter due to improved training time. In
the third case, filtering may still be desirable, depending on the trade-off between
performance (e.g. accuracy) and efficiency (e.g. train time): a small reduction in
accuracy may be acceptable if build time reduces by an order of magnitude. We
address the task of how best to select a subset of dimensions for high dimensional
data so that we can speed up and possibly improve HC2 on high dimensional
MTSC problems.
Detecting the best subset of dimensions is not a straightforward problem,
since the number of combinations to consider increases exponentially with the
number of dimensions. Selection is also made more complex by the fact that
the objective function used to assess a set of features may not generalise well
to unseen data. Furthermore, since the primary reason for filtering the dimensions
is improving the efficiency of the classifier, dimension selection strategies
themselves need to be fast.

Currently we have the channel selection algorithms describe in [1,2] in aeon. It would
be great to include those in [3] and further work. This project will involve
experimental evaluation in addition to implementing
algorithms. We can co-ordinate the experiments with the candidate through our HPC
facilities.

1. Implement a channel selection wrapper for the aeon toolkit (see [#1270](https://github.com/aeon-toolkit/aeon/issues/1270))
2. Explore alternative ways of selecting channels after scoring (e.g. forward selection)
3. Use a fast classifier that can find train estimates through e.g. bagging and avoid the cross validation
4. Research, implement and evaluate alternative channel selection algorithms

##### References
[1] Dhariyal, B. et al. Fast Channel Selection for Scalable Multivariate Time
Series Classification. AALTD, ECML-PKDD, Springer, 2021
[2] Dhariyal, B. et al. Scalable Classifier-Agnostic Channel Selection
    for Multivariate Time Series Classification", DAMI, 2023
[3] Ruiz, A.P., Bagnall, A. Dimension Selection Strategies for Multivariate
   Time Series Classification with HIVE-COTEv2.0. AALTD,ECML-PKDD 2022.
   (https://doi.org/10.1007/978-3-031-24378-3_9)

### Visualisation

#### 1. Explainable AI with the shapelet transform.

Mentors: TonyBagnall ({user}`TonyBagnall`) and David Guijo-Rubio
({user}`dguijo`)

This project will focus on explainable AI for time series classification (TSC) [1],
specifically the family of algorithms based on shapelets [2]. Shapelets are small sub
 certain shape of heartbeat, perhaps a short irregularity, might be useful in
predicting the medical condition. We will look at the shapelet transform classifier
[3]. This finds a large set of shapelets from the training data and uses them to
build a classifier. We want to develop tools to help us visualise the output of the
search for good shapelets to help explain why predictions are made. This project is
not tied to a specific data set. It is to develop tools to help any user of the
toolkit.  It will involve learning about aeon and making contributions to open
source toolkits, familiarisation with the shapelet code and the development of a
visualisation tool to help relate shapelets back to the training data. An outline
for the project is

1. Familiarisation with open source, aeon and the visualisation module. Make
contribution for a good first issue.
2. Understand the shapelet transfer algorithm, engage in ongoing discussions
for possible improvements, run experiments to create predictive models for a test data set
3. Design and prototype visualisation tools for shapelets, involving a range
of summary measures and visualisation techniques, including plotting shapelets on training data, calculating frequency, measuring similarity between
4. Debug, document and make PRs to merge contributions into the aeon toolkit.

[1] Bagnall, A., Lines, J., Bostrom, A., Large, J. and Keogh, E. The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances. Data Mining and Knowledge Discovery, Volume 31, pages 606–660, (2017)
[2] Ye, L., Keogh, E. Time series shapelets: a novel technique that allows accurate, interpretable and fast classification. Data Min Knowl Disc 22, 149–182 (2011). https://doi.org/10.1007/s10618-010-0179-5
[3] Lines, L., Davis, L., Hills, J. and Bagnall, A. A shapelet transform for time series classification, KDD '12: Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining (2012) https://doi.org/10.1145/2339530.2339579

### Regression

#### 1. Adapt forecasting regressors to time series extrinsic regression.

Mentors: TonyBagnall ({user}`TonyBagnall`) and David Guijo-Rubio
({user}`dguijo`)

Forecasting is often reduced to regression through the application of a sliding
window. This is a large research field that is distinct to time series extrinsic
regression, where each series is assumed to be independent. This is more of a
research project to investigate what techniques are used in forecasting for
regression based forecasting and to compare them to the time series specific
algorithms in aeon. This project would require further working up with the mentors.


#### 2. Adapt HIVE-COTE for regression

Mentors: TonyBagnall ({user}`TonyBagnall`) and David Guijo-Rubio
({user}`dguijo`)

HIVE-COTE [1] is a state of the art classifier. Adapting it for regression is an
ongoing research project for which we would welcome collaborators. Ongoing, this
needs working up.


### Documentation

#### 1. Improve automated API documentation

Mentors: Matthew Middlehurst ({user}`MatthewMiddlehurst`)

##### Description

`aeon` uses `sphinx` and `numpydoc` to generate API documentation from docstrings.
Many of the docstrings are incomplete or missing sections, and could be improved to
make the API documentation more useful. The goal of this project is to generally
improve the API documentation. A specific goal is to automatically generate links to
examples which use the function/class, similar to the `scikit-learn` documentation.
The way this is achieved is up to the mentee, but should include a new section in the
relevant API page. I.e., the API page for
`aeon.transformers.collection.convolution_based.Rocket` should have a section called
"Examples" which links to the examples which use the class (such as the Rocket
notebook).

##### Project Stages

1. Learn about `aeon` best practices and project documentation.
2. Familiarise with `sphinx` documentation generation and `numpydoc` docstring
standards.
3. Improve the API documentation for a few classes/functions and go through the Pull
Request and review process.
4. Implement a function or improve the API template to automatically generate links
to examples which use the function/class.
5. The main bulk of work is done, but the API documentation is vast and can always be
improved! If time allows, continue to enhance the API documentation through individual
docstrings, API landing page and template improvements at the mentees discretion.

##### Project Outcomes

Success of the project will be assessed by the quality of the documentation produced
and engagement with the project and the `aeon` community. Automatically generating
links to examples is the primary goal.
