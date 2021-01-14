# category-generator
Codebase for "Generating CCG Categories" (AAAI 2021)

## Requirements
* `python`: 3.6.10
* [`dynet`](https://dynet.readthedocs.io/en/latest/python.html): 2.0.0
* [`antu`](https://github.com/AntNLP/antu/tree/feature/dynamic_oracle_adaptation): 0.0.4

## Results
|       | description         |   dev   | test    | Download |
| ----  |    ----             |   ----  |  ----   |  ----    |
| CC    | category classifier | 95.00   |   95.14      |  [`link`](https://drive.google.com/file/d/1qGzrfsiFawjqZg0u9L0OQrk5G_O3a7gx/view?usp=sharing) (218M)   |
| CG    | category generator  | 95.09   |   95.27      |  [`link`](https://drive.google.com/file/d/1gJ7SdqN-OUsjRrWGXY7yHKfXaorGppRn/view?usp=sharing) (222M)   |
| CGNG2 | category generator with deterministic 2-gram oracle|   95.21      |   95.38      |  [`link`](https://drive.google.com/file/d/1y3w2hPn9QO3gpTdOTyPaIhhmkghC-5NP/view?usp=sharing) (222M)   |


## Training
```bash
$ python train.py --name AAAI2020 (your experiment name) --gpu 0 (your gpu id) --model generator (which model, classifier or generator)
``` 
Before training, please make sure everything is all you want in configs/connfig.cfg. 
Such as the filepath of the pre-trained embeddings
(we use [`Turian embedding`](http://metaoptimize.s3.amazonaws.com/hlbl-embeddings-ACL2010/hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt.gz), 84.3MB),
model storage path and whether to use different oracles.

## Testing
```bash
$ python test.py --gpu 0 (your gou id) --model (run which model)
``` 

## Contact
For questions and usage issues, please contact yfliu.antnlp@gmail.com .


