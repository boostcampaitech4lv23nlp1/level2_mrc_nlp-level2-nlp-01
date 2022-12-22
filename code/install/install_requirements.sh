#!/bin/bash
### install requirements for pstage3 baseline
# pip requirements
pip install torch==1.13
pip install datasets==2.7.1
pip install transformers==4.25.1
pip install tqdm
pip install pandas
pip install scikit-learn

# faiss install (if you want to)
pip install faiss-gpu
