import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

class SimilarContentNN:
    def __init__(self, batch_size=32, min_vote_count=100, min_popularity=15):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_vote_count = min_vote_count
        self.min_popularity = min_popularity

        print(f"Using device: {self.device}")

        # Load the BERT model and tokenizer for feature embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.model.eval()

        # Load Sentence Transformer for overview embeddings
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)

        # Weights for Movies: overview, genres, keywords, content_category.
        self.m_w_overview = 0.5
        self.m_w_genres = 0.3
        self.m_w_keywords = 0.1
        self.m_w_catg = 0.1
        
        # Weights for TV shows: overview, genres, content_category.
        self.s_w_overview = 0.5
        self.s_w_genres = 0.3
        self.s_w_catg = 0.2

    def create_sentence_embeddings(self, texts):
        """Create embeddings using Sentence Transformer (for overview)."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Creating sentence embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            with torch.no_grad():
                embeddings = self.sentence_transformer.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def create_bert_embeddings(self, texts):
        """Create embeddings using BERT for other text features."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Creating BERT embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)

    def _project_embeddings(self, embeddings, target_dim):
        """Project embeddings to a lower dimension using PCA."""
        if embeddings.shape[1] <= target_dim:
            return embeddings
        pca = PCA(n_components=target_dim)
        return pca.fit_transform(embeddings)

    def process_data(self, df):
        """Main method to process content data."""
        print("Processing data...")
        filtered_data = self._preprocess_data(df)
        movies_df, tv_df = self._process_content_data(filtered_data)
        unified_df = self._combine_embeddings(movies_df, tv_df)
        self.df = unified_df
        print("Processing complete!")

    def _preprocess_data(self, df):
        """Filter and split data."""
        filtered_df = df[(df['vote_count'] >= self.min_vote_count) & 
                        (df['popularity'] >= self.min_popularity) & 
                        (df['vote_average'] >= 6)].copy()
        
        if len(filtered_df) == 0:
            raise ValueError("No data left after filtering")
        return filtered_df

    def _process_content_data(self, df):
        """Process movies and TV shows separately."""
        movies_df = df[df['type'] == "Movie"].copy()
        tv_df = df[df['type'] == "TV"].copy()
        
        processed_movies = self._process_movies(movies_df)
        processed_tv = self._process_tv_shows(tv_df)
        return processed_movies, processed_tv

    def _process_movies(self, movies_df):
        """Process movie-specific data and create embeddings."""
        movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split(", ") if isinstance(x, str) else [])
        movies_df['keywords'] = movies_df['keywords'].apply(lambda x: x.split(", ") if isinstance(x, str) else [])
        movies_df["genres_text"] = movies_df["genres"].apply(lambda x: " ".join(x * 2))
        movies_df["keywords_text"] = movies_df["keywords"].apply(lambda x: " ".join(x))
        movies_df["catg_text"] = movies_df["content_category"].astype(str)
        
        movies_df["embedding"] = self._create_movie_embeddings(movies_df)
        return movies_df

    def _process_tv_shows(self, tv_df):
        """Process TV show-specific data and create embeddings."""
        tv_df['genres'] = tv_df['genres'].apply(lambda x: x.split(", ") if isinstance(x, str) else [])
        tv_df["genres_text"] = tv_df["genres"].apply(lambda x: " ".join(x * 2))
        tv_df["catg_text"] = tv_df["content_category"].astype(str)
        
        tv_df["embedding"] = self._create_tv_embeddings(tv_df)
        return tv_df

    def _create_movie_embeddings(self, df):
        """Create and combine embeddings for movies."""
        overview_emb = self.create_sentence_embeddings(df["overview"].tolist())
        genre_emb = self._project_embeddings(self.create_bert_embeddings(df["genres_text"].tolist()), 384)
        keyword_emb = self._project_embeddings(self.create_bert_embeddings(df["keywords_text"].tolist()), 384)
        catg_emb = self._project_embeddings(self.create_bert_embeddings(df["catg_text"].tolist()), 384)
        
        combined = (
            self.m_w_overview * normalize(overview_emb) +
            self.m_w_genres * normalize(genre_emb) +
            self.m_w_keywords * normalize(keyword_emb) +
            self.m_w_catg * normalize(catg_emb)
        )
        return list(normalize(combined))

    def _create_tv_embeddings(self, df):
        """Create and combine embeddings for TV shows."""
        overview_emb = self.create_sentence_embeddings(df["overview"].tolist())
        genre_emb = self._project_embeddings(self.create_bert_embeddings(df["genres_text"].tolist()), 384)
        catg_emb = self._project_embeddings(self.create_bert_embeddings(df["catg_text"].tolist()), 384)
        
        combined = (
            self.s_w_overview * normalize(overview_emb) +
            self.s_w_genres * normalize(genre_emb) +
            self.s_w_catg * normalize(catg_emb)
        )
        return list(normalize(combined))

    def _combine_embeddings(self, movies_df, tv_df):
        """Combine processed movies and TV shows."""
        unified_df = pd.concat([movies_df, tv_df], ignore_index=True)
        self.embeddings_matrix = np.vstack(unified_df["embedding"].values)
        return unified_df.reset_index(drop=True)

