#get a prompt
#use sentence trasformer to analyse the prompt
#maybe fine tune it
#create results for genre,rating etc.
# I suppose I would need to get embeddings so I can compare with the main model and get recommendation

## Start with type. Movie or Tv. Get AI to generate some prompts and see if it can classify them correctly.There should be a algorithm. If the prompt is vague and not specifying a genre or any details but just movie then user should get the most popular movies in the dataset. Like a top-down approach. The more specific the prompt the more specific the recommendation.
# Maybe you can establish a score for the prompt. If the score isn't too high then the prompt is vague recommend something generic. If the score is high then recommend something specific.
## need a mapping for genres. Get all the genres from the dataset. See if the model can produce the expected outputs. Like for the prompt "I want to watch a sad movie about an old couple -> Romance/Drama" 
# for overviews try to understand how the embeddings are created

# Workflow: Get a prompt. Create embeddings for the prompt. compare embeddings with the main model. Get recommendations based on the comparison. Classify for type then genre then  lastly use overivew or keywords to get the final recommendation
# There are filtering information. Such as rating,popularity,year etc. Maybe for the later versions you can look for director or actor.
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class PromptNN:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Initialize the SentenceTransformer model.
        self.model_st = SentenceTransformer(model_name)
        # Define prototype sentences.
        self.prototype_movie_list = [
            "This is a feature film meant for theatrical release with cinematic visuals and storytelling.",
            "A movie is designed for a big screen experience with dramatic narratives and high production values.",
            "This is a film that offers a complete story, usually lasting around two hours and shown in theaters."
        ]
        self.prototype_tv_list = [
            "This is a television show designed for episodic storytelling, often with multiple seasons.",
            "A TV show is created for broadcast on television or streaming services with recurring episodes.",
            "This is a series made for episodic viewing, where each episode builds on the previous ones."
        ]
        # Compute and store normalized prototype embeddings.
        self.prototype_movie_emb = self._compute_normalized_embedding(self.prototype_movie_list)
        self.prototype_tv_emb = self._compute_normalized_embedding(self.prototype_tv_list)
    
    def _compute_normalized_embedding(self, texts: list) -> np.ndarray:
        # Compute embeddings, average, and normalize.
        embeddings = self.model_st.encode(texts)
        avg_emb = np.mean(embeddings, axis=0, keepdims=True)
        norm_emb = avg_emb / np.linalg.norm(avg_emb)
        return norm_emb

    def _encode_and_normalize(self, prompt: str) -> np.ndarray:
        emb = self.model_st.encode([prompt])
        norm_emb = emb / np.linalg.norm(emb)
        return norm_emb

    def classify(self, prompt: str) -> str:
        """
        Classifies a prompt as 'Movie' or 'TV' by comparing the prompt
        embedding with precomputed prototype embeddings.
        """
        prompt_emb = self._encode_and_normalize(prompt)
        sim_movie = cosine_similarity(prompt_emb, self.prototype_movie_emb)[0][0]
        sim_tv = cosine_similarity(prompt_emb, self.prototype_tv_emb)[0][0]
        
        # Debug prints for similarity scores.
        print(f"Cosine similarity with Movie prototype: {sim_movie:.3f}")
        print(f"Cosine similarity with TV prototype: {sim_tv:.3f}")
        
        return "Movie" if sim_movie >= sim_tv else "TV"

# Example usage:
if __name__ == "__main__":
    nn_classifier = PromptNN()
    prompt = "I want to watch a sad movie about an old couple."
    predicted_type = nn_classifier.classify(prompt)
    print("Predicted content type:", predicted_type)