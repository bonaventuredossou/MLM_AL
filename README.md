# MLM_AL

- Run `python tokenizer_150k/build_tokenizer.py` to build the SP-BPE tokenizer with `150k` codes. Or use the existing ones.
- I am currently using `XLM-R Large`. You can run for another version `small` or `base`. Please check [here](https://github.com/castorini/afriberta/tree/6cacc453f3a99a6f902174e8e7f8dd6184c1794f/mlm_configs) for their respective configs and adapt accordingly using my current `large.yml` config file
- Specify the model size [here](https://github.com/bonaventuredossou/MLM_AL/blob/master/source/trainer.py#L29)
- run `sbatch/bash emnlp22_all.sh` or directly `python active_learning.py`
