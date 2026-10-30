# Movie Recommendation System

An intelligent movie recommendation system built using **Neural Collaborative Filtering (NCF)** to provide personalized movie suggestions based on user preferences and rating behaviour.

The project combines Generalized Matrix Factorization (GMF) and a Multi-Layer Perceptron (MLP) to learn user–movie interactions and predict ratings. It also addresses new-user and new-movie cold-start problems through hybrid recommendation techniques.

Built using the MovieLens 32M dataset, the system includes an offline model-training pipeline, an online recommendation engine, an interactive Gradio interface, and an on-demand model retraining mechanism.

## Project Overview

The system is divided into two main phases:

### 1. Offline Phase — Model Training

The offline pipeline prepares the MovieLens dataset and trains the Neural Collaborative Filtering model.

The process includes:

- **Dataset Analysis:** Examining users, movies, ratings, genres, missing values and duplicate interactions.
- **Data Cleaning:** Removing duplicate records, handling missing values and correcting data types.
- **Chronological Data Splitting:** Dividing each user's rating history into 80% training, 10% validation and 10% testing data.
- **ID Encoding:** Converting original user and movie IDs into continuous numerical indices. Mappings are created using training data only to prevent data leakage.
- **NCF Model Training:** Learning user and movie representations through the GMF and MLP components.
- **Model Evaluation:** Evaluating rating predictions using RMSE and MAE.
- **Model Saving:** Saving the trained model, ID mappings and configuration details for online use.

### 2. Online Phase — Recommendation Engine

The online phase loads the trained model and uses it to generate personalized recommendations.

It supports three user scenarios:

- Existing MovieLens users with historical rating data.
- Existing application users with saved profiles and feedback.
- New users who register and provide their initial preferences.

Based on the user's model availability and rating history, the system selects an appropriate recommendation method.

## Recommendation Techniques

### Neural Collaborative Filtering

The core recommendation model combines two learning paths:

**Generalized Matrix Factorization (GMF)**

Learns direct interactions between user and movie latent embeddings through element-wise multiplication.

**Multi-Layer Perceptron (MLP)**

Learns complex, nonlinear relationships by passing concatenated user and movie embeddings through multiple neural network layers.

The outputs of both paths are concatenated and passed through a final layer to predict the user's rating for a movie.

### New User Cold-Start Handling

A new user does not initially have a learned user embedding in the trained NCF model.

To generate recommendations before retraining, the system collects:
- 2–3 preferred genres.
- Initial ratings for 5–10 popular movies.

It then builds a user preference vector from the trained movie embeddings and calculates a weighted score using:

- 50% — Similarity to movies the user has rated.
- 30% — Preferred genre matching.
- 20% — Movie popularity.

This enables personalized recommendations even before the user's ratings are incorporated into NCF training.

### New Movie Cold-Start Handling

Movies that are not represented in the trained NCF model cannot receive normal NCF predictions because they have no learned movie embeddings.

The system handles these movies using a separate fallback score:

- 60% — Genre matching.
- 40% — Popularity.

Up to two positions in the Top-10 recommendation list are reserved for eligible model-unseen movies.

### Hybrid Recommendation Strategy

The system combines NCF predictions with content-based genre matching and popularity-based scoring to handle situations where direct collaborative predictions are unavailable.

## User Interaction and Feedback

The interactive application is built using Gradio.

Users can:

- Register or log in.
- Select preferred movie genres.
- Rate initial movies during onboarding.
- Request personalized Top-10 recommendations.
- Filter recommendations by genre.
- Rate recommended movies.
- View ratings saved in their application profile.
- Log out and return later.

User profiles, selected genres and application-generated ratings are stored in an SQLite database.

## Continuous Learning Through Retraining

The system supports on-demand model retraining using ratings collected through the application.

The retraining process:

1. Exports application ratings into MovieLens-compatible CSV format.
2. Loads the original rating dataset.
3. Cleans and chronologically splits the data.
4. Adds eligible application ratings to the training data.
5. Rebuilds user and movie ID mappings.
6. Trains a fresh NCF model using the updated data.
7. Evaluates the model using validation RMSE, test RMSE and test MAE.
8. Saves the updated model checkpoint.
9. Reloads the model and updates the recommendation engine.

After retraining, newly registered users included in the training data receive learned NCF user embeddings and can obtain NCF-based recommendations.

## Dataset

**MovieLens 32M**

The project uses the MovieLens 32M dataset containing:

| Dataset component | Count |
|---|---:|
| User ratings | 32,000,204 |
| Users | 200,948 |
| Movies in catalogue | 87,585 |

The dataset provides user–movie ratings, movie titles, genres and timestamps.

Dataset source: [MovieLens — GroupLens Research](https://grouplens.org/datasets/movielens/32m/)

## Model Performance

The offline NCF model was trained for three epochs using an NVIDIA Tesla T4 GPU.

**Training configuration**

- Model: Neural Collaborative Filtering (GMF + MLP)
- Embedding dimension: 32
- Trainable parameters: 17,446,689
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Mean Squared Error
- Batch size: 4,096

**Validation performance**

| Epoch | Validation RMSE |
|---|---:|
| 1 | 0.8465 |
| 2 | 0.8325 |
| 3 | 0.8215 |

**Final offline test results**

| Metric | Result |
|---|---:|
| Test RMSE | **0.8380** |
| Test MAE | **0.6340** |

These test metrics were calculated on 3,102,624 of the 3,114,491 test interactions whose movies were represented in the training mapping. The remaining interactions involved movies unseen during training and are handled separately through the new-movie fallback in the online recommendation pipeline.

During an example online retraining run with eight application ratings, the test RMSE was 0.8379 and the test MAE was 0.6370.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | NCF model development, training and inference |
| Pandas | Dataset processing and data management |
| NumPy | Numerical operations and recommendation scoring |
| SQLite | User profiles, genres and rating storage |
| Gradio | Interactive user interface |
| Google Colab | Development and execution environment |
| NVIDIA Tesla T4 GPU | Accelerated model training |

## Running the Project

The project is developed to run in Google Colab using Google Drive for dataset and model storage.

### 1. Prepare the dataset

Download the MovieLens 32M dataset and place the following files in your Google Drive's `MyDrive` folder:

- `movies.csv`
- `ratings.csv`

### 2. Install dependencies

Run the following in a Colab cell if the packages are not already available:

```bash
pip install torch pandas numpy matplotlib seaborn gradio
```

### 3. Run the offline phase

Open and execute the offline training notebook.

This performs preprocessing, chronological splitting, ID encoding, NCF model training and evaluation.

The trained checkpoint must be saved to:

```text
/content/drive/MyDrive/ncf_movie_recommender.pth
```

Make sure the offline save path matches the `model_path` configured in the online phase.

### 4. Run the online phase

Execute the online recommendation engine code after the offline model has been saved.

The engine loads the checkpoint, movie catalogue and SQLite database, then prepares the recommendation system.

### 5. Launch the Gradio interface

Execute the Gradio UI code in the same runtime after the online engine and its required classes, functions and configuration have been initialized.

The interface can then be launched to register users, generate recommendations, collect ratings and trigger retraining.

**Note:** The Gradio configuration uses `share=True`, which creates a shareable application link. This is a project prototype, not a production authentication or deployment setup.

## Important Implementation Notes

- A user's new ratings are saved immediately, but NCF model weights are updated only when retraining is executed.
- New users initially receive cold-start recommendations and switch to NCF-based recommendations after their IDs and ratings are incorporated into training.
- Model-unseen movies are not necessarily recently released movies; they are movies missing from the current NCF training mapping.
- Retraining creates a fresh model rather than incrementally fine-tuning the previous checkpoint.
- App ratings for user–movie pairs already present in the original ratings dataset are skipped during the current retraining merge.

## Future Improvements

- Evaluate recommendation ranking using Precision@K, Recall@K and NDCG@K on a clearly defined candidate set.
- Improve new-item recommendations using richer movie metadata.
- Explore more advanced hybrid recommendation techniques.
- Improve deployment, authentication and user-data management.
- Experiment with larger models and additional training configurations.

## References

1. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). *Neural Collaborative Filtering*. Proceedings of the 26th International Conference on World Wide Web. https://doi.org/10.1145/3038912.3052569

2. Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 5(4). https://doi.org/10.1145/2827872

3. Son, L. H. (2016). *Dealing with the new user cold-start problem in recommender systems: A comparative review*. Information Systems, 58. https://doi.org/10.1016/j.is.2014.10.001

4. Deldjoo, Y., et al. (2019). *Movie Genome: Alleviating New Item Cold Start in Movie Recommendation*. User Modeling and User-Adapted Interaction, 29(2). https://doi.org/10.1007/s11257-019-09221-y

---

**Developed as an end-to-end movie recommendation project combining neural collaborative filtering, hybrid cold-start handling and feedback-driven model retraining.**
