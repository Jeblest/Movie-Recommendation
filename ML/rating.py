# Decide on an algorithm which will evaluate the rating score. Like threshold etc. 
# Model stays the same you store the multipliers and calculate the new embeddings on the fly.
# Do you hold different models for each user or is it like a overarching model and additional personalized small infos which the model uses
#Store the rating information in the database as a user table. Each row will hold the multiplier for that movie, where the multiplier comes from(genre,type etc.)
#Process:
#User rates a movie
#recommended movies from the model for that movie gets updated(I'm not sure how the update will happen. I'm thinking adjusting the combined embeddings matrix)

def ratingMultiplier(rating,content):
    #you get the information from the db.
    #get the similar movies from the model for that content
    #when
    pass