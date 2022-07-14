### Preprocessing Flan To JSONL Format
the script `examples/few_shot/prompt_to_lm_data.py` just reads the Flan tasks  (using `examples/few_shot/tasks.py`) and converts them to jsonl format and the write file structure.  (see `/private/home/sshleifer/fairseq-py/flan_streaming` for the result).

### Usage
```
PYTHONPATH='.' python examples/few_shot/prompt_to_lm_data.py --split valid --save-dir flan_streaming
```
### Result

```
head -n 1 flan_streaming/valid/00/flan__aeslc_10templates.jsonl

{"text": "This is the content of an email: Elizabeth-  I am working on the Federal Estate Tax Return for your mother's Estate and I notice that I don't have any expenses in my file. Do you have a list of expenses such funeral expenses, etc.? Ellen H. Arthur  Hodes, Ulman, Pessin & Katz, PA  901 Dulaney Valley Road, Suite 400  Towson, MD 21204  (410)769-6146  NOTICE: The information contained in this electronic mail transmission is intended by Hodes, Ulman, Pessin & Katz, P.A. for the use of the named individual or entity to which it is directed and may contain information that is privileged or otherwise confidential. It is not intended for transmission to, or receipt by, anyone other than the named addressee (or a person authorized to deliver it to the named addressee). It should not be copied or forwarded to any unauthorized persons. If you have received this electronic mail transmission in error, please delete it from your system without copying or forwarding it, and notify the sender of the error by reply email or by calling Hodes, Ulman, Pessin & Katz, P.A. at (410) 938-8800 or 1-800-276-0806 so that our address record can be corrected. \nWhat was the subject line for this email? Estate of Joan M. Sager"}
```


If you pass `--no-streaming` to the script, you can get the data in a format for `LanguageModelingTask` (Not streaming).




### Sample Break Mode
Yeah:
the default dataloader logic (`--sample-break-mode none`) for both `LanguageModelingTask` and `StreamingLanguageModelingTask` is
to combine "samples" until such that each sample is  of length `--tokens-per-sample`.

For example, if we assume 10 tokens per sample and that each word is 1 token:
(note these are all 14 words)

Dataset:
```
{"text": "Donald Trump was the 44th President of The United States. Options Yes No. Yes"}
{"text": "Barack Obama was the 43rd President of The United States. Options Yes No. Yes"}
{"text": "George Bush was the 43rd President of The United States. Options Yes No. Yes"}
```

would be fed to the model as 5 samples
```
[Donald Trump was the 44th President of The United States. ,
Options Yes No. Yes. Barack Obama was the 43rd President,
of The United States. Options Yes No. Yes George Bush,
was the 43rd President of The United States. Options Yes ,
No. Yes PAD PAD PAD PAD PAD PAD PAD PAD,
]
```


This seems  suboptimal because we don't guarantee that the model can attend to the prompt when it is predicting the label word.


With `--sample-break-mode complete-doc`:
we truncate each example to the first N tokens and then just feed that to the model with pads if needed.
In this example, we'd get:
```
[44th President of The United States. Options Yes No. Yes
43rd President of The United States. Options Yes No. Yes
43rd President of The United States. Options Yes No. Yes
]
```
This again seems bad, but if `tokens-per-sample `were longer than the average prompt, as is usually the case for FLAN tasks, it would make more sense than the first way.
