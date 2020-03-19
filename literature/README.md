# Literature

### Contents

* [1. Survey Papers](#1-survey-papers)
* [2. Attack Papers](#2-attack-papers)
	* [2.1 Text Classification Attacks](#21-text-classification-attacks)
	* [2.2 Natural Language Inference Attacks](#22-natural-language-inference-attacks) 
	* [2.3 Multi-task Attacks](#23-multi-task-attacks)
	* [2.4 General Attacks](#24-general-attacks)
* [3. Defense Papers](#3-defense-papers)
* [4. Robustness Papers](#4-robustness-papers)



## 1. Survey Papers

1. **Towards a Robust Deep Neural Network in Texts: A Survey**.
*Wenqi Wang, Lina Wang, Benxiao Tang, Run Wang, Aoshuang Ye*. arXiv 2020. [[pdf](https://arxiv.org/pdf/1902.07285.pdf)]
1. **Adversarial Attacks and Defenses in Images, Graphs and Text: A Review**.
*Han Xu, Yao Ma, Haochen Liu, Debayan Deb, Hui Liu, Jiliang Tang, Anil K. Jain*. arXiv 2019. [[pdf](https://arxiv.org/pdf/1909.08072.pdf)]
1. **Adversarial Attacks on Deep Learning Models in Natural Language Processing: A Survey**.
*Wei Emma Zhang, Quan Z. Sheng, Ahoud Alhazmi, Chenliang Li*. arXiv 2019. [[pdf](https://arxiv.org/pdf/1901.06796.pdf)]


## 2. Attack Papers

### 2.1 Text Classification Attacks
1. **Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers**.
*Ji Gao, Jack Lanchantin, Mary Lou Soffa, Yanjun Qi*. **IEEE SPW 2018**. `score` `blind` [[pdf](https://ieeexplore.ieee.org/document/8424632)] [[code](https://github.com/QData/deepWordBug)]
1. **HotFlip: White-Box Adversarial Examples for Text Classification**.
*Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou*. **ACL 2018**. `gradient` [[pdf](https://www.aclweb.org/anthology/P18-2006)] [[code](https://github.com/AnyiRao/WordAdver)]

### 2.2 Natural Language Inference Attacks
1. **Probing Neural Network Understanding of Natural Language Arguments**.
*Timothy Niven, Hung-Yu Kao*. **ACL 2019**. `score` [[pdf](https://www.aclweb.org/anthology/P19-1459.pdf)] [[code&data](https://github.com/IKMLab/arct2)]
1. **Breaking NLI Systems with Sentences that Require Simple Lexical Inferences**.
*Max Glockner, Vered Shwartz, Yoav Goldberg*. **ACL 2018**. `blind` [[pdf](https://www.aclweb.org/anthology/P18-2103)] [[dataset](https://github.com/BIU-NLP/Breaking_NLI)]

### 2.3 Multi-task Attacks
1. **Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment**.
*Di Jin, Zhijing Jin, Joey Tianyi Zhou, Peter Szolovits*. **AAAI-20**. `score` [[pdf](https://arxiv.org/pdf/1907.11932v4)] [[code](https://github.com/wqj111186/TextFooler)]

### 2.4 General Attacks
1. **PAWS: Paraphrase Adversaries from Word Scrambling**.
*Yuan Zhang, Jason Baldridge, Luheng He*. **NAACL-HLT 2019**. `blind` [[pdf](https://www.aclweb.org/anthology/N19-1131)] [[dataset](https://g.co/dataset/paws)]
1. **Generating Natural Adversarial Examples**.
*Zhengli Zhao, Dheeru Dua, Sameer Singh*. **ICLR 2018**. `decision` [[pdf](https://arxiv.org/pdf/1710.11342.pdf)] [[code](https://github.com/zhengliz/natural-adversary)]
1. **Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency**.
*Shuhuai Ren, Yihe Deng, Kun He, Wanxiang Che*. **ACL 2019**. `score` [[pdf](https://www.aclweb.org/anthology/P19-1103.pdf)] [[code](https://github.com/JHL-HUST/PWWS/)]
1. **Generating Fluent Adversarial Examples for Natural Languages**.
*Huangzhao Zhang, Hao Zhou, Ning Miao, Lei Li*. **ACL 2019**. `gradient` `score` [[pdf](https://www.aclweb.org/anthology/P19-1559)] [[code](https://github.com/LC-John/Metropolis-Hastings-Attacker)]
1. **Generating Natural Language Adversarial Examples**.
*Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang*. **EMNLP 2018**. `score` [[pdf](https://www.aclweb.org/anthology/D18-1316)] [[code](https://github.com/nesl/nlp_adversarial_examples)]
1. **Interpretable Adversarial Perturbation in Input Embedding Space for Text**.
*Sato, Motoki, Jun Suzuki, Hiroyuki Shindo, and Yuji Matsumoto*. **IJCAI 2018**. `gradient` [[pdf](https://arxiv.org/pdf/1805.02917.pdf)] [[code](https://github.com/aonotas/interpretable-adv)]
1. **Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems**.
*Steffen Eger, Gözde Gül ¸Sahin, Andreas Rücklé, Ji-Ung Lee, Claudia Schulz, Mohsen Mesgar, Krishnkant Swarnkar, Edwin Simpson, Iryna Gurevych*. **NAACL-HLT 2019**. `score` [[pdf](https://www.aclweb.org/anthology/N19-1165)] [[code&data](https://github.com/UKPLab/naacl2019-like-humans-visual-attacks)]
1. **TEXTBUGGER: Generating Adversarial Text Against Real-world Applications**.
*Jinfeng Li, Shouling Ji, Tianyu Du, Bo Li, Ting Wang*. **NDSS 2019**. `gradient` `score` [[pdf](https://arxiv.org/pdf/1812.05271.pdf)]
1. **Universal Adversarial Triggers for Attacking and Analyzing NLP**.
*Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, Sameer Singh*. **EMNLP-IJCNLP 2019**. `gradient` [[pdf](https://arxiv.org/pdf/1908.07125.pdf)] [[code](https://github.com/Eric-Wallace/universal-triggers)] [[website](http://www.ericswallace.com/triggers)]

## 3. Defense Papers
1. **Learning to Discriminate Perturbations for Blocking Adversarial Attacks in Text Classification**.
*Yichao Zhou, Jyun-Yu Jiang, Kai-Wei Chang, Wei Wang*. **EMNLP-IJCNLP 2019**. [[pdf](https://www.aclweb.org/anthology/D19-1496.pdf)] [[code](https://github.com/joey1993/bert-defender)]
1. **Combating Adversarial Misspellings with Robust Word Recognition**.
*Danish Pruthi, Bhuwan Dhingra, Zachary C. Lipton*. **ACL 2019**. [[pdf](https://www.aclweb.org/anthology/P19-1561.pdf)] [[code](https://github.com/danishpruthi/adversarial-misspellings)]

## 4. Robustness Papers
1. **Robustness Verification for Transformers**.
*Zhouxing Shi, Huan Zhang, Kai-Wei Chang, Minlie Huang, Cho-Jui Hsieh*. **ICLR 2020**. [[pdf](https://arxiv.org/pdf/2002.06622.pdf)] [[code](https://github.com/shizhouxing/Robustness-Verification-for-Transformers)]
1. **Certified Robustness to Adversarial Word Substitutions**.
*Robin Jia, Aditi Raghunathan, Kerem Göksel, Percy Liang*. **EMNLP-IJCNLP 2019**. [[pdf](https://www.aclweb.org/anthology/D19-1423.pdf)] [[code](https://github.com/robinjia/certified-word-sub)]
