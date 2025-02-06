import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import difflib

def get_content_type(df, type_val):
    """Filter content by type (Movie or TV)."""
    return df[df['type'] == type_val]

def find_closest_match(df, title):
    """Finds the closest title match from the dataframe."""
    matches = difflib.get_close_matches(title, df['title'], n=1)
    return matches[0] if matches else None

def get_recommendations(model, title, type_val, top_n=10, alpha=0.9, mmr_lambda=0.7):
    """Get content recommendations based on similarity and ratings."""
    # First filter the dataframe by type
    type_df = get_content_type(model.df, type_val)
    
    # Find the closest match within the filtered dataframe
    matched_title = find_closest_match(type_df, title)
    
    if not matched_title:
        print(f"Content '{title}' not found in {type_val} database")
        return
    
    # Get the global index of the target content
    target_global_idx = model.df[model.df['title'] == matched_title].index[0]
    target_embedding = model.embeddings_matrix[target_global_idx].reshape(1, -1)
    
    # Get indices for all content of the same type
    same_type_indices = type_df.index.to_numpy()
    
    # Calculate similarities only for content of the same type
    type_embeddings = model.embeddings_matrix[same_type_indices]
    similarities = cosine_similarity(type_embeddings, target_embedding).flatten()
    
    # Get normalized ratings for the same type content
    normalized_ratings = type_df['vote_average'].to_numpy() / 10
    
    # Calculate final scores
    final_scores = alpha * similarities + (1 - alpha) * normalized_ratings
    
    # Create a mapping from position in type_df to position in final_scores
    target_type_idx = np.where(same_type_indices == target_global_idx)[0][0]
    
    # Get sorted indices excluding the target
    local_indices = np.argsort(final_scores)[::-1]
    similar_local_indices = [idx for idx in local_indices if idx != target_type_idx][:top_n]
    
    # Map back to global indices
    similar_global_indices = same_type_indices[similar_local_indices]
    
    # Get recommendations
    recommendations = model.df.iloc[similar_global_indices]
    
    print(f"\nTop {top_n} {type_val} recommendations for '{matched_title}':")
    for i, (_, content) in enumerate(recommendations.iterrows(), 1):
        local_idx = similar_local_indices[i-1]
        print(f"{i}. {content['title']} "
              f"(Score: {final_scores[local_idx]:.3f}, "
              f"Rating: {content['vote_average']}, "
              f"Type: {content['type']}, "
              f"Category: {content['content_category']}, "
              f"Popularity: {content['popularity']:.1f})")