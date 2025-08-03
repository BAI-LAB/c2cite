This repository contains the code for the paper “C$^2$-Cite: Contextual-Aware Citation Generation for \\ Attributed Large Language Models”. The project is based on the open-source repository"[TUDB-Labs/MoE-PEFT](https://github.com/TUDB-Labs/MoE-PEFT)". C$^2$-Cite is a model that can answer the questions with citation markers.
## File description
- **config**: Including the configurations of training or evaluating
- **c2cite/backends**: Some backend tools for GMoE.
- **c2cite/common**: The implementation of Transformer architecture.
- **c2cite/models**: The implementation of some series of Transformer-based models.
- **c2cite/tasks**: The implementation of datasets.
- **c2cite.py** The start file of this project.
## Environment Requirements
 - python3=3.11
 - pytorch >= 2.1.2
 - Other dependencies, See ```requirements.txt```
## Quick Start
### STEP 1: Download Base models
 - [Llama-3-8B-inst]
### STEP 2: Downlaod training datasets
To get Training dataset proposed in paper "Towards Faithful and Robust LLM Specialists for Evidence-Based Question-Answering", you can download [SynSciQA](https://github.com/EdisonNi-hku/Robust_Evidence_Based_QA) here. And please put SynSciQA.json, SynSciQA+.json, SynSciQA++.json in ./dataset/SynSciQA
### STEP 3: Download evaluation datasets
We evaluate our model and baselines using [ALCE](https://github.com/princeton-nlp/ALCE). To get Evaluate datasets, please run 
```bash 
bash download_test_data.sh
```
### STEP 4: Start training 
Replace the **[base model]** and the **[train/evaluate config]** below with the directory of base model and the configuration in Folder "config".
``````python
python c2cite.py --dir ./checkpoint --log_file ./logs --verbose --seed 42 --attn_impl eager --base_model [base model] --config [train/evaluate config] --device cuda:0
``````
### STEP 5: Conduct evaluation
After training process, we can conduct the evaluation step with the command below:
``````python
python c2cite.py --dir ./checkpoint --log_file ./logs --verbose --seed 42 --attn_impl eager --base_model [base model] --config [train/evaluate config] --device cuda:0 --evaluate
``````
***Note***:   **Do not** change the information in the **train config** after training step, or it won't find the right adapter.

