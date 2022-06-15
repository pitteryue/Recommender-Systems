# Recommender systems based on graph embedding techniques: A review
As a pivotal tool to alleviate the information overload problem, recommender systems aim to predict user’s preferred items from millions of candidates by analyzing observed user-item relations. As for alleviating the sparsity and cold start problems encountered by recommender systems, researchers generally resort to employing side information or knowledge in recommendation as a strategy for uncovering hidden (indirect) user-item relations, aiming to enrich observed information (or data) for recommendation. However, in the face of the high complexity and large scale of side information and knowledge, this strategy largely relies for efficient implementation on the scalability of recommendation models. Not until after the prevalence of machine learning did graph embedding techniques be a recent concentration, which can efficiently utilize complex and large-scale data. In light of that, equipping recommender systems with graph embedding techniques has been widely studied these years, appearing to outperform conventional recommendation implemented directly based on graph topological analysis (or resolution). As the focus, this article systematically retrospects graph embedding-based recommendation from embedding techniques for bipartite graphs, general graphs and knowledge graphs, and proposes a general design pipeline of that. In addition, after comparing several representative graph embedding-based recommendation models with the most common-used conventional recommendation models on simulations, this article manifests that the conventional models can still overall outperform the graph embedding-based ones in predicting implicit user-item interactions, revealing the comparative weakness of graph embedding-based recommendation in these tasks. To foster future research, this article proposes constructive suggestions on making a trade-off between graph embedding-based recommendation and conventional recommendation in different tasks, and puts forward some open questions.

Article link: https://ieeexplore.ieee.org/abstract/document/9772660

**Since a review article, the evaluation experiments conducted in it were partially implemented on selected public codes of benchmarks from github. References are presented as follows:**
1. https://github.com/clhchtcjj/BiNE was partially referenced by BiNE_graph.py, BiNE_graph_utils.py, and BiNE_lsh.py.
2. https://github.com/coreylynch/pyFM and https://github.com/rixwew/pytorch-fm were partially referenced by FM.py and FM2.py.
3. https://github.com/mayukh18/reco/tree/master/reco/recommender was partially referenced by FunkSVD.py.
4. https://github.com/guoyang9/NCF was partially referenced by NCF.py.
5. https://github.com/adamzjw/Probabilistic-matrix-factorization-in-Python and https://github.com/xuChenSJTU/PMF was partially referenced by PMF.py.
6. https://github.com/erickrf/autoencoder/tree/master/src was partially referenced by SAE.py.
7. https://github.com/wuxiyu/transE was partially referenced by TransE.py.
