# OffensEval 2019 (SemEval 2019 - Task 6)
## Identification of offensive language

#### Please visit <https://competitions.codalab.org/competitions/20011#learn_the_details-overview> for further information
#### The results were presented in the paper SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval) which can be found at <https://arxiv.org/abs/1903.08983>. See the team 'Vadym' for the results that can be achieved. Place: 10 out of 800.

1. Please download FastText Embeddings: <https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip> 
and extract it to the project folder.
2. Download the data and extract it to the project folder.
2. Install all required packages: `pip3 install -r requirements.txt`. It is recommended to use a virtual environment.
3. The project folder should contain:
    - 'training-v1' directory
    - 'model_output' directory
4. Tensorboard can be run by executing the command `tensorboard --logdir ./model_output/`
5. Run `Model.py`

See also `Paper.pdf` for more details
