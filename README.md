## Neural Networks for Natural Language Processing 
This repository contains some useful functions, classes and codes snippets, which provide some general frameworks, logits, pipeline or structure for implementing some state-of-the-art deep learning natural language processing models.


## Structure
public scripts:
There are two codebases:
* `./tf`: the tensorflow codebase
* `./torch`: the pytorch codebase

Within each codesbase:
* `./modeling.py`: contains some popular models/layers, such as transformer block, bert/albert, lstm layer, etc.. 
* `./loss.py`: different loss, softmax with label smoothing, focal loss, crf loss, etc..
* `./optimization.py`: optional optimization script.
* `./tokenization.py`: for text tokenization.
* some util scripts

demo training scripts:
* train_glue_baseline.py: which is sigle layer bilstm baseline model for GLUE task.
> python train_glue_baseline.py --data_dir data/ --task_name cola --vocab_file model/vocab.txt --output_dir debug --max_seq_length 8 --do_train=1 --do_eval=1 --train_batch_size 1 --eval_batch_size 1 --train_step 4 --config_file model/config.json


other scripts:
* download_glue_data.py: download_glue_data.
> python download_glue_data.py --data_dir data --tasks all
* glue_utils.py: for GLUE processing.
* lamb_optimizer.py: lamb_optimizer.


## Details

### Modeling


### Loss
* `softmax`: softmax with optional label smoothing
* `crf`: crf loss and decoding
* `am_softmax`: Addictive angular softmax
* `contrastive_loss`: pairvise loss
* `sigmoid_focal_crossentropy`: focal loss function for multiple anchors or sequence of tokens