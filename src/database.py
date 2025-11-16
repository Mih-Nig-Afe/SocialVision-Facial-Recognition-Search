"""
Database management for face embeddings and metadata
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from src.logger import setup_logger
from src.config import get_config

logger = setup_logger(__name__)
config = get_config()


class FaceDatabase:
    """Local JSON-based face database"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize face database
        
        Args:
            db_path: Path to database file
        """
        self.db_path = Path(db_path or config.LOCAL_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load_database()
        logger.info(f"Initialized FaceDatabase at {self.db_path}")
    
    def _load_database(self) -> Dict[str, Any]:
        """Load database from file"""
        try:
            if self.db_path.exists():
                with open(self.db_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading database: {e}")
        
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "faces": [],
            "metadata": {}
        }
    
    def _save_database(self) -> bool:
        """Save database to file"""
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"Database saved successfully to {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving database: {e}", exc_info=True)
            return False
    
    def add_face(
        self,
        embedding: List[float],
        username: str,
        source: str,
        image_url: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a face embedding to database
        
        Args:
            embedding: Face embedding (typically 128 or 512 dimensions)
            username: Instagram username
            source: Source of image (profile_pic, post, story, reel)
            image_url: URL of the image
            metadata: Additional metadata
        
        Returns:
            True if successful
        """
        try:
            # Validate embedding
            if not embedding:
                logger.error("Empty embedding provided")
                return False
            
            if not isinstance(embedding, list):
                logger.error(f"Embedding must be a list, got {type(embedding)}")
                return False
            
            embedding_dim = len(embedding)
            if embedding_dim == 0:
                logger.error("Embedding has zero dimensions")
                return False
            
            logger.info(f"Adding face: username={username}, source={source}, embedding_dim={embedding_dim}")
            
            face_record = {
                "id": len(self.data["faces"]),
                "embedding": embedding,
                "username": username,
                "source": source,
                "image_url": image_url,
                "added_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            self.data["faces"].append(face_record)
            
            if self._save_database():
                logger.info(f"Successfully added face for user {username} from {source} (embedding_dim={embedding_dim})")
                return True
            else:
                logger.error("Failed to save database after adding face")
                # Remove the face from memory if save failed
                self.data["faces"].pop()
                return False
        
        except Exception as e:
            logger.error(f"Error adding face: {e}", exc_info=True)
            return False
    
    def get_all_embeddings(self) -> np.ndarray:
        """Get all embeddings as numpy array"""
        try:
            embeddings = [face["embedding"] for face in self.data["faces"]]
            return np.array(embeddings) if embeddings else np.array([]).reshape(0, 128)
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return np.array([]).reshape(0, 128)
    
    def get_face_by_id(self, face_id: int) -> Optional[Dict]:
        """Get face record by ID"""
        try:
            for face in self.data["faces"]:
                if face["id"] == face_id:
                    return face
        except Exception as e:
            logger.error(f"Error getting face by ID: {e}")
        return None
    
    def get_faces_by_username(self, username: str) -> List[Dict]:
        """Get all faces for a username"""
        try:
            return [face for face in self.data["faces"] if face["username"] == username]
        except Exception as e:
            logger.error(f"Error getting faces by username: {e}")
            return []
    
    def search_similar_faces(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.6,
        top_k: int = 50
    ) -> List[Dict]:
        """
        Search for similar faces
        
        Args:
            query_embedding: Query face embedding
            threshold: Similarity threshold
            top_k: Number of top results to return
        
        Returns:
            List of similar faces sorted by similarity
        """
        try:
            if not self.data["faces"]:
                return []
            
            embeddings = self.get_all_embeddings()
            
            # Calculate distances
            distances = np.linalg.norm(
                embeddings - query_embedding,
                axis=1
            )
            
            # Convert distances to similarity scores (0-1)
            similarities = 1 / (1 + distances)
            
            # Filter by threshold
            valid_indices = np.where(similarities >= threshold)[0]
            
            # Sort by similarity
            sorted_indices = valid_indices[np.argsort(-similarities[valid_indices])][:top_k]
            
            # Build results
            results = []
            for idx in sorted_indices:
                face = self.data["faces"][idx]
                results.append({
                    **face,
                    "similarity_score": float(similarities[idx])
                })
            
            logger.info(f"Found {len(results)} similar faces")
            return results
        
        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            usernames = set(face["username"] for face in self.data["faces"])
            sources = {}
            for face in self.data["faces"]:
                source = face["source"]
                sources[source] = sources.get(source, 0) + 1
            
            return {
                "total_faces": len(self.data["faces"]),
                "unique_users": len(usernames),
                "sources": sources,
                "created_at": self.data["created_at"]
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def clear_database(self) -> bool:
        """Clear all data from database"""
        try:
            self.data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "faces": [],
                "metadata": {}
            }
            self._save_database()
            logger.warning("Database cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

