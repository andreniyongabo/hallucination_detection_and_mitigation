# Hallucination detection and mitigation
This repository contains step-by-step guidelines on how to reproduce the codes for hallucination detection and mitigation techiniques that are based on this [internship plan](https://docs.google.com/document/d/1VGA763JBhVghCJYH2LNtkqBc3Wg9Msl4E_SLXV8ZtXA/edit) and the detailed summary of the internship progress can also be found [here](https://docs.google.com/document/d/1gg0HHv-YTs-MRpxZrei0oECJih0tg4129-9p721cONs/edit#). We use eng-kin pair from flores devetest but the codes can be easily adapted to any other pair from flores by just replacing the source and target language codes.

## How to use
- Clone this repo to your home directory as follows:
  
  `git clone https://github.com/andreniyongabo/hallucination_detection_and_mitigation.git`
- Change your directory to `fairseq-py`:
  
  `cd fairseq-py`
- And install it:
  
  `pip install --editable .`
  
  `pip install hydra-core --upgrade --pre`
  
  `pip install fairscale`

### halucination detection
to desing a hallucination detection we need an [annotated dataset](https://docs.google.com/spreadsheets/d/1MoG7WJNnDlO-C4-HQ-SPxEjO7IKGRAcD3pjs5gpXQA4/edit?usp=sharing), `sentence score`, `similarity score`, and `alti score`.
- Getting sentence score:
  
  `cd ..`
  
  `bash translate.sh`
  
- Getting similarity score:

  `bash laser_similarity_score.sh`
  
- Getting alti score:

  `cd alti/src`
  
  `bash source_contribution_beam.sh`
  
- Preparing the data for detection:

  `cd ../..`
  
  `python create_csv_for_eval.py`

- Then follow the steps in [this notebook]() on the `hallucination detection` section
### hallucination mitigation
For mitigation we need `source sentences`, `reference sentences`, `hypotesis sentences`, `sentence score` and `similarity score`. Let still use the above [annotated dataset](https://docs.google.com/spreadsheets/d/1MoG7WJNnDlO-C4-HQ-SPxEjO7IKGRAcD3pjs5gpXQA4/edit?usp=sharing) for mitigation where `source sentences` and `reference sentences` are already given.
- First translate source sentences to get all candidates (hypothesis) in the beam and their sentence scores (considering you are still at `halucination_detection_and_mitigation` directory) :
  
  `bash get_beam_candidates.sh`
  
- Get the similarity score:

  `bash source_contribution_beam.sh`
  
- Get new hypothesis by reranking the beam based on the similarity score and then both on the sentence score and similarity score:

  `python beam_reranking.py`
  
- Compare the bleu scores:
  
  `bash get_bleu_score.sh`
  
