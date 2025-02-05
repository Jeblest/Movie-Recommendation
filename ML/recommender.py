import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import difflib

def get_content_type(df,type):
    return df[df['type'] == type]

def find_closest_match(df, title):
        """Finds the closest title match from the dataframe."""
        matches = difflib.get_close_matches(title, df['title'], n=1)
        return matches[0] if matches else None

def get_recommendations(model, title, type_val, top_n=10, alpha=0.9):
    """Get content recommendations based on similarity and ratings."""
    filtered_df = get_content_type(model.df, type_val)
    matched_title = find_closest_match(filtered_df, title)
    
    if not matched_title:
        print(f"Content '{title}' not found in database")
        return

    # Get target content details
    target_idx = filtered_df.index[filtered_df['title'] == matched_title][0]
    target_embedding = model.embeddings_matrix[target_idx].reshape(1, -1)
    
    # Calculate similarities with all content
    similarities = cosine_similarity(model.embeddings_matrix, target_embedding).flatten()
    
    # Combine with ratings
    normalized_ratings = model.df['vote_average'] / 10
    final_scores = alpha * similarities + (1 - alpha) * normalized_ratings.values
    
    # Get indices of top similar content
    indices = np.argsort(final_scores)[::-1]
    similar_indices = [idx for idx in indices if idx != target_idx][:top_n]

    # Get recommendations
    recommendations = model.df.iloc[similar_indices]
    
    print(f"\nTop {top_n} recommendations for '{matched_title}':")
    for i, (_, content) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {content['title']} "
              f"(Score: {final_scores[content.name]:.3f}, "
              f"Rating: {content['vote_average']}, "
              f"Type: {content['type']}, "
              f"Category: {content['content_category']}, "
              f"Popularity: {content['popularity']:.1f})")