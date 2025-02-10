import numpy as np
from btfr.massfuncs import get_GSMF_Adams, get_GSMF_ELPETRO
from BAM import AbundanceMatch, proxies
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation import ValidationRunner
from ili import utils

training_data = np.load("training_data.npy")

print(training_data.shape)

loader = NumpyLoader(x=training_data[0, :].reshape(-1, 1), theta=training_data[1:, :].T)

# train a model to infer x -> theta. save it as toy/posterior.pkl
trainer = InferenceRunner.load(
  backend = 'sbi', engine='NPE',                # Choose a backend and inference engine (here, Neural Posterior Estimation)
  prior = utils.Uniform(low=[10, 1], high=[15, 50]),    # Define a prior 
  # Define a neural network architecture (here, Mixture Density Network)
  nets = [utils.load_nde_sbi(engine='NPE', model='mdn', hidden_features=10, num_components=4)],
  out_dir="models"
)

posterior_ensemble, summaries = trainer(loader)

print("Training done")
print(summaries)