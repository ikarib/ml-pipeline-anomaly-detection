# What Did Not Work

## 1. Using a lower autoencoder threshold
At first I tried a more aggressive threshold so the autoencoder would catch more anomalies. Recall improved, but alert volume rose too much and precision deteriorated.

## 2. Treating the autoencoder as the “headline” model
A neural model sounds attractive in a portfolio repo, but on this dataset it was not the best business tradeoff. Keeping it in the project makes sense as a comparison, not as the winner.

## 3. Overcomplicating the feature set
I considered adding many more rolling windows and lag features. That quickly made the project noisier without clearly improving the story. I kept the features compact and explainable instead.

## 4. Presenting synthetic data as if it were real
That would make the repo read as more impressive on the surface, but less trustworthy. I decided it was better to state clearly that the data are synthetic and show how they were generated.

