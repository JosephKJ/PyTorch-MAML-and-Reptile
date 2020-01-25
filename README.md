# MAML and Reptile in PyTorch

This repository includes the Sine wave experiment with MAML and Reptile.

### Tested on
```shell script
Python 3.7.4
PyTorch 1.3.1
```

### Run
```shell script
python main.py --run=MAML
python main.py --run=Reptile
```

### Results

```shell script
Loss after 30000 iterations
---------------------------
MAML: 0.058
REPTILE: 0.048
```
![MAML](https://github.com/JosephKJ/MAML-and-Reptile/results/maml.png "MAML")
![Reptile](https://github.com/JosephKJ/MAML-and-Reptile/results/reptile.png "Reptile")



#### Adapted from
[John Schulman's GIST](https://gist.github.com/joschu/f503500cda64f2ce87c8288906b09e2d#file-reptile-sinewaves-demo-py)