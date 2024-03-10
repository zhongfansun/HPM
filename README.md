# Breaking the Barrier Between Pre-training and Fine-tuning: A Hybrid Prompting Model for Knowledge-Based VQA

Official implementation for the MM'23 paper. The corresponding preprocessed data and preprocessing methods will be continuously updated.

## Prerequisites
* python==3.7
* pytorch==1.10.0

### Model Training:
```python
python main.py --name unifer_mlmitm --dataset OKVQA --cfg './cfgs/ok-vqa.yaml' --gpu 1 --task 'itm' --vlmodel 'vinvl'
```

### Citation:
If you found this repo helpful, please consider cite the following paper :+1: :
```ruby
@inproceedings{sun2023breaking,
  title={Breaking the Barrier Between Pre-training and Fine-tuning: A Hybrid Prompting Model for Knowledge-Based VQA},
  author={Sun, Zhongfan and Hu, Yongli and Gao, Qingqing and Jiang, Huajie and Gao, Junbin and Sun, Yanfeng and Yin, Baocai},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4065--4073},
  year={2023}
}
```
## Acknowledgements
This code is heavily based on [UnifER](https://github.com/guoyang9/UnifER) and [DPT](https://github.com/CCIIPLab/DPT).