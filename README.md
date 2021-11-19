# Machine Learning Based Abstract Reasoning Applied to Visual IQ Tests

This project explores the use of machine learning to solve visual IQ tests such as [Raven's Progressive Matrices](https://en.wikipedia.org/wiki/Raven%27s_Progressive_Matrices). The underlying code we've used for this project is from the following research paper: [Abstract Reasoning with Distracting Features](http://arxiv.org/abs/1912.00569). Along with further testing and validation of the results and methods of the research paper, we have added more features (GUI) as well as potential methods of improvements (Vision Transformer, MADDPG).


<div width="20%", height="20%", align="center">
   <img src="https://github.com/zkcys001/distracting_feature/blob/master/git_images/LEN.png"><br><br>
</div>


# Dependencies

**Important**
* Python 2.7
* PyTorch
* CUDA and cuDNN


# Usage

## Benchmarking

```
python main.py --net <model name> --datapath <path to the dataset> --rl False --typeloss False
```

# Citation

```
@inproceedings{zheng2019abstract,
    title={Abstract Reasoning with Distracting Features},
    author={Kecheng Zheng and Zheng-jun Zha and Wei Wei},
    booktitle={Advances in Neural Information Processing Systems},
    year={2019}}
}
```

# Acknowledgement


* [Wild Relational Network](https://github.com/Fen9/WReN)


