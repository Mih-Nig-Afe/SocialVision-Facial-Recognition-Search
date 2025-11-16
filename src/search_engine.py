"""
Search engine for finding similar faces
"""

import numpy as np
from typing import List, Dict, Optional
from src.logger import setup_logger
from src.database import FaceDatabase
from src.face_recognition_engine import FaceRecognitionEngine

logger = setup_logger(__name__)


class SearchEngine:
    """Facial recognition search engine"""
    
    def __init__(self, database: FaceDatabase):
        """
        Initialize search engine
        
        Args:
            database: FaceDatabase instance
        """
        self.database = database
        self.face_engine = FaceRecognitionEngine()
        logger.info("Initialized SearchEngine")
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.6,
        top_k: int = 50
    ) -> List[Dict]:
        """
        Search for similar faces by embedding
        
        Args:
            query_embedding: Query face embedding
            threshold: Similarity threshold
            top_k: Number of top results
        
        Returns:
            List of similar faces with metadata
        """
        try:
            results = self.database.search_similar_faces(
                query_embedding,
                threshold=threshold,
                top_k=top_k
            )
            
            logger.info(f"Search returned {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error searching by embedding: {e}")
            return []
    
    def search_by_image(
        self,
        image: np.ndarray,
        threshold: float = 0.6,
        top_k: int = 50
    ) -> Dict:
        """
        Search for similar faces by image
        
        Args:
            image: Input image (BGR format)
            threshold: Similarity threshold
            top_k: Number of top results
        
        Returns:
            Dictionary with search results for each detected face
        """
        try:
            logger.info("Starting face detection in search_by_image...")
            # Detect faces
            face_locations = self.face_engine.detect_faces(image)
            logger.info(f"Face detection completed. Found {len(face_locations)} face(s)")
            
            if not face_locations:
                logger.warning("No faces detected in query image")
                return {"faces": [], "total_matches": 0}
            
            logger.info("Extracting face embeddings...")
            # Extract embeddings
            embeddings = self.face_engine.extract_face_embeddings(image, face_locations)
            logger.info(f"Extracted {len(embeddings)} embedding(s)")
            
            if len(embeddings) == 0:
                logger.warning("No embeddings extracted - DeepFace may not be available")
                # Return empty results but indicate faces were detected
                return {
                    "faces": [{
                        "face_index": i,
                        "location": loc,
                        "matches": []
                    } for i, loc in enumerate(face_locations)],
                    "total_matches": 0
                }
            
            # Search for each face
            results = {
                "faces": [],
                "total_matches": 0
            }
            
            logger.info(f"Searching database for {len(embeddings)} face(s)...")
            for i, embedding in enumerate(embeddings):
                if len(embedding) == 0:
                    logger.warning(f"Empty embedding for face {i}, skipping search")
                    continue
                    
                face_results = self.search_by_embedding(
                    embedding,
                    threshold=threshold,
                    top_k=top_k
                )
                
                results["faces"].append({
                    "face_index": i,
                    "location": face_locations[i] if i < len(face_locations) else None,
                    "matches": face_results
                })
                results["total_matches"] += len(face_results)
            
            logger.info(f"Image search completed with {results['total_matches']} total matches")
            return results
        
        except Exception as e:
            logger.error(f"Error searching by image: {e}", exc_info=True)
            return {"faces": [], "total_matches": 0}
    
    def get_unique_usernames(self, search_results: Dict) -> List[str]:
        """
        Extract unique usernames from search results
        
        Args:
            search_results: Search results dictionary
        
        Returns:
            List of unique usernames
        """
        try:
            usernames = set()
            
            for face_result in search_results.get("faces", []):
                for match in face_result.get("matches", []):
                    usernames.add(match["username"])
            
            return sorted(list(usernames))
        
        except Exception as e:
            logger.error(f"Error extracting usernames: {e}")
            return []
    
    def get_results_by_username(self, search_results: Dict) -> Dict[str, List]:
        """
        Group search results by username
        
        Args:
            search_results: Search results dictionary
        
        Returns:
            Dictionary mapping usernames to their matches
        """
        try:
            results_by_user = {}
            
            for face_result in search_results.get("faces", []):
                for match in face_result.get("matches", []):
                    username = match["username"]
                    if username not in results_by_user:
                        results_by_user[username] = []
                    results_by_user[username].append(match)
            
            return results_by_user
        
        except Exception as e:
            logger.error(f"Error grouping results by username: {e}")
            return {}
    
    def get_top_usernames(
        self,
        search_results: Dict,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Get top usernames by match count and average similarity
        
        Args:
            search_results: Search results dictionary
            top_k: Number of top results
        
        Returns:
            List of top usernames with statistics
        """
        try:
            results_by_user = self.get_results_by_username(search_results)
            
            user_stats = []
            for username, matches in results_by_user.items():
                avg_similarity = np.mean([m["similarity_score"] for m in matches])
                user_stats.append({
                    "username": username,
                    "match_count": len(matches),
                    "avg_similarity": float(avg_similarity),
                    "sources": list(set(m["source"] for m in matches))
                })
            
            # Sort by match count and similarity
            user_stats.sort(
                key=lambda x: (x["match_count"], x["avg_similarity"]),
                reverse=True
            )
            
            return user_stats[:top_k]
        
        except Exception as e:
            logger.error(f"Error getting top usernames: {e}")
            return []

