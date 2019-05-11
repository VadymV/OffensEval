# OffensEval
## Identification of offensive language using Python3 and Tensorflow

1. Please download FastText Embeddings: <https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip> 
and extract it to the project folder.
2. Install all required packages: `pip3 install -r requirements.txt`
3. The project folder should contain:
    - 'training-v1' directory
    - 'model_output' directory
4. Tensorboard can be run by executing the command `tensorboard --logdir ./model_output/`