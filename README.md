# LogicMD
The official implementation of our paper "Interpretable Multimodal Misinformation Detection with Logic Reasoning", was accepted by Finding of ACL 23 (4/4/4 for excitement and soundness). More detailed documents will be uploaded in a few weeks. If you have some questions related to this paper, please email me (liuhui3-c@my.cityu.edu.hk).

# Dataset 
You can download the dataset and the related pre-processing files in this link \url{https://portland-my.sharepoint.com/:u:/g/personal/liuhui3-c_my_cityu_edu_hk/EYR-45i16q9EivlGM1ZCe9cBrPsuOjr8O9fziihKJPLIoA?e=2KVCFW}.
# Instruction to run our code
Our models are implemented in models/rule_detection.py. To reproduce the results of experiments, you can use train_sarcasm.py for the Sarcasm dataset and train_two.py for Weibo and Twitter datasets. 

Since I used comet package in these files, you may remove codes related to this package or just replace my api keys with yours. After that, you can run sarcasm.sh, twitter.sh, and weibo.sh directly. 

Note (2023/12/08): I found some problems in codes, try to fix them in a fewer weeeks.

