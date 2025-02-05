class UserInteractionCollector:
    def __init__(self):
        self.user_interactions_db = {
            "explicit_feedback": [],
            "implicit_feedback": [],
            "search_data": [],
            "session_data": []
        }
    
    def collect_explicit_feedback(self, user_id, content_id, interaction_type, timestamp):
        feedback = {
            "user_id": user_id,
            "content_id": content_id,
            "timestamp": timestamp,
            "interaction": {
                "type": interaction_type,  # rating, review, like/dislike
                "value": None,
                "text": None
            }
        }
        
        if interaction_type == "rating":
            feedback["interaction"]["value"] = rating_value
        elif interaction_type == "review":
            feedback["interaction"]["text"] = review_text
            # Store sentiment analysis result
            feedback["interaction"]["sentiment"] = self.analyze_sentiment(review_text)
        
        self.user_interactions_db["explicit_feedback"].append(feedback)

    def collect_implicit_feedback(self, user_id, content_id, action, timestamp):
        feedback = {
            "user_id": user_id,
            "content_id": content_id,
            "timestamp": timestamp,
            "action": {
                "type": action,  # watch_time, click, hover, complete_viewing
                "duration": None,
                "context": None
            }
        }
        self.user_interactions_db["implicit_feedback"].append(feedback)

    def collect_search_data(self, user_id, search_query, results_clicked, timestamp):
        search_interaction = {
            "user_id": user_id,
            "timestamp": timestamp,
            "search": {
                "query": search_query,
                "results_shown": [],
                "results_clicked": results_clicked,
                "session_duration": None
            }
        }
        self.user_interactions_db["search_data"].append(search_interaction)

class DataPreprocessor:
    def __init__(self):
        self.training_datasets = {}
    
    def prepare_rating_matrix(self, explicit_feedback):
        """Convert raw ratings into a user-item matrix"""
        user_item_matrix = defaultdict(dict)
        
        for feedback in explicit_feedback:
            if feedback["interaction"]["type"] == "rating":
                user_id = feedback["user_id"]
                item_id = feedback["content_id"]
                rating = feedback["interaction"]["value"]
                user_item_matrix[user_id][item_id] = rating
        
        return user_item_matrix

    def prepare_search_training_data(self, search_data):
        """Prepare search queries and successful results for training"""
        search_training_data = []
        
        for search in search_data:
            query = search["search"]["query"]
            successful_results = search["search"]["results_clicked"]
            
            # Create positive examples from clicked results
            for result in successful_results:
                search_training_data.append({
                    "query": query,
                    "content": result,
                    "label": 1  # positive example
                })
            
            # Create negative examples from non-clicked results
            shown_but_not_clicked = set(search["search"]["results_shown"]) - set(successful_results)
            for result in shown_but_not_clicked:
                search_training_data.append({
                    "query": query,
                    "content": result,
                    "label": 0  # negative example
                })
        
        return search_training_data

    def create_content_similarity_data(self, user_sessions):
        """Create training data for content similarity based on user behavior"""
        similarity_pairs = []
        
        for session in user_sessions:
            watched_items = session["watched_content"]
            
            # Items watched in the same session are considered related
            for i in range(len(watched_items)):
                for j in range(i + 1, len(watched_items)):
                    similarity_pairs.append({
                        "item1": watched_items[i],
                        "item2": watched_items[j],
                        "similarity_score": self.calculate_session_similarity_score(
                            watched_items[i], watched_items[j], session
                        )
                    })
        
        return similarity_pairs
    
class RecommenderTrainingSystem:
    def __init__(self):
        self.collector = UserInteractionCollector()
        self.preprocessor = DataPreprocessor()
        self.training_schedule = {
            "full_retrain": "weekly",
            "incremental_update": "daily"
        }
    
    def process_new_interactions(self, new_interactions):
        """Process incoming user interactions in real-time"""
        # Store raw interaction data
        for interaction in new_interactions:
            self.collector.process_interaction(interaction)
            
        # Update necessary online models
        self.update_online_models(new_interactions)
    
    def prepare_training_batch(self, start_date, end_date):
        """Prepare a batch of training data for model updates"""
        raw_data = self.collector.get_interactions_in_range(start_date, end_date)
        
        training_data = {
            "ratings": self.preprocessor.prepare_rating_matrix(
                raw_data["explicit_feedback"]
            ),
            "search": self.preprocessor.prepare_search_training_data(
                raw_data["search_data"]
            ),
            "content_similarity": self.preprocessor.create_content_similarity_data(
                raw_data["session_data"]
            )
        }
        
        return training_data

    def validate_data_quality(self, training_data):
        """Implement validation checks for training data"""
        quality_metrics = {
            "completeness": self.check_data_completeness(training_data),
            "consistency": self.check_data_consistency(training_data),
            "distribution": self.analyze_data_distribution(training_data)
        }
        
        return quality_metrics