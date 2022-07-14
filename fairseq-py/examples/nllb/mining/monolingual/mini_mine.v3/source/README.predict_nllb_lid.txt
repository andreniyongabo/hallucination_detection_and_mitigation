Example to use predict_nllb_lid.py;

```
cat original-file.txt | ./predict_nllb_lid.py --model-date last > lid-predictions.txt
cat original-file.txt | ./predict_nllb_lid.py --print-prob --model-date last > lid-predictions.txt
```

or a model from a specific date:

```
cat original-file.txt | ./predict_nllb_lid.py --model-date 2021-10-12 > lid-predictions.txt
```

If you want to specificy the model (and corresponding best thresholds):

```
cat original-file.txt | ./predict_nllb_lid.py --model /large_experiments/nllb/mmt/lidruns/2021-10-05-16-36-multifilter/result/model.8.8.bin --thresholds /large_experiments/nllb/mmt/lidruns/2021-10-07-10-55-optim-threshold/result/thresholds_2.npy > lid-predictions.txt
```
