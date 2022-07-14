# laser teacher-student iterative mining

### context

The goal of iterative mining using the teacher-student approach is to see how much improvement can be achieved training a student model using data which the student previously mined. The teacher-student method works by taking text pairs and using regression loss to try to best match the student and teacher's representations. This method is favourable since it can dramatically reduce training time, and has been shown to produce promising results.

This code uses [Hydra](https://hydra.cc/) to specify all parameters in a single config: `config.yaml`. Parameters can also be added or overwritten when calling the program.

### preprocessing

Given an input directory containing bitexts, source-target languages, and a number of chosen corpora, the intput texts are automatically gathered and any preprocessing (e.g. moses tokenization) or sampling which are specified in the config will be peformed. A data config is then generated to be sent as input to the data preparation/binarization pipeline under `examples/nllb/modeling`.

cmd: `python teacher_student.py mode=preprocess`

### training and evaluation

All training params as input to fairseq can be specified in the `config.yaml` file under `train_config`, or each can be specified via the command line. For example: `train_config.dropout=0.1`. Each time a checkpoint is saved, fairseq will automatically evaluate it and save the results before continuing training. During this process, the laser encoder will first be extracted from the saved checkpoint and then sent for evaluation. By default the evaluation script measures the xsim error rate on flores101 using all src languages which are specified in the config under `src_langs`. 

cmd: `python teacher_student.py mode=train [train_config.lr=0.0005]`


### ah-hoc evaluation

It's also possible to ad-hoc evaluate a particular saved checkpoint. The encoder will be extracted from the checkpoint and then passed to an evaluation script. 

cmd: `python teacher_student.py mode=eval checkpoint=[checkpoint_file.pt]`
