## Tensorflow-nlp
This repository contains some useful tensorflow coding, which provide a general frame and structure for implementing some state-of-the-art natural language processing models.


## Structure
public scripts:
|-- modeling.py: contains some popular models/layers, such as bert/albert, crf layer, lstm layer, etc.. 

|-- optimization.py: optimization script.

|-- tokenization.py: for text tokenization.

demo training scripts:

|-- train_glue_baseline.py: which is sigle layer bilstm baseline model for GLUE task.
> python train_glue_baseline.py --data_dir data/ --task_name cola --vocab_file model/vocab.txt --output_dir debug --max_seq_length 8 --do_train=1 --do_eval=1 --train_batch_size 1 --eval_batch_size 1 --train_step 4 --config_file model/config.json


other scripts:
|-- download_glue_data.py: download_glue_data.
> python download_glue_data.py --data_dir data --tasks all

|-- glue_utils.py: for GLUE processing.

|-- lamb_optimizer.py: lamb_optimizer.


