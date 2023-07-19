# LogicMD
The offical implementation of our paper "Interpretable Multimodal Misinformation Detection with Logic Reasoning", accepted by Finding of ACL 23. More detail documents will be uploded in a few weeks. If you have some questions related to this paper, please email me (liuhui3-c@my.cityu.edu.hk).

# Dataset 
You can download the dataset and the related pre-processing files in this link \url{https://portland-my.sharepoint.com/:u:/g/personal/liuhui3-c_my_cityu_edu_hk/EYR-45i16q9EivlGM1ZCe9cBrPsuOjr8O9fziihKJPLIoA?e=2KVCFW}.
# Instruction to run our code
Our models are implemented in models/rule_detection.py. To reproduce the results of experiments, you can use train_sarcasm.py for Sarcasm dataset and train_two.py for Weibo and Twitter datasets. Since I used comet package in these files, you may remove codes related to this package or just replace the my api keys with yous.
