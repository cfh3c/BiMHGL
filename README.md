# Code of Bi-Layer Multimodal Hypergraph Learning
Bi-Layer Multimodal Hypergraph Learning is the advanced version of [Multimodal Hypergraph Learning (Multi-HGL)](https://github.com/cfh3c/Multi-HGL)

## Brief Explanation
* run_CV_gridsearch.m: an entrance for transductive learning, evaluation and inference
* BiHG_learning2.m: a core part for bi-Layer multimodal hypergraph learning (someone can use or advances it for other tasks, e.g., including the initialization of edge weights, fixing W, and optimizing main variables/parameters (f, W, g and F) beyond levels)
* preprocess*.m: pre-processing codes for data (we were informed that the data was sensitive, so you can refer to the codes to pre-process your data)
* mPara.mStarExp, mPara.mLamda, mPara.mMu, mPara.mProbSigmaWeight, mPara.Alpha, mPara.mLamda2, and mPara.mMu2 are main hyper-parameters (Please refer to the paper). mPara.mStarExp and mPara.mMu2 are much more important.

## Citing Multi-HGL

If you find Multi-HGL code useful in your research, please consider citing:

    @article{ji2018cross,
      title={Cross-modality microblog sentiment prediction via bi-layer multimodal hypergraph learning},
      author={Ji, Rongrong and Chen, Fuhai and Cao, Liujuan and Gao, Yue},
      journal={IEEE Transactions on Multimedia},
      volume={21},
      number={4},
      pages={1062--1075},
      year={2018},
      publisher={IEEE}
      }
