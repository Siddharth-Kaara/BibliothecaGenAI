import logging
from typing import Type, List, Dict, Any, Optional, Tuple, ClassVar
import re
import sqlalchemy 
from sqlalchemy.ext.asyncio import AsyncConnection
from pydantic import BaseModel, Field 

from langchain_core.tools import BaseTool

# Add additional imports for better matching
from rapidfuzz import fuzz, process as rapidfuzz_process
from functools import partial

from app.db.connection import get_async_db_connection

logger = logging.getLogger(__name__)

# --- Input Schema ---
class HierarchyResolverInput(BaseModel):
    name_candidates: List[str] = Field(description="A list of potential hierarchy names (e.g., branch names, library names) mentioned by the user.")
    # organization_id: str = Field(description="The specific organization ID to scope the hierarchy search.") # Removed - Tool uses internal context
    # Optional: Add a threshold if you want the agent to control it, otherwise use a default
    # min_score_threshold: int = Field(default=85, description="Minimum confidence score (0-100) to consider a match valid.")

# --- Tool Implementation ---
class HierarchyNameResolverTool(BaseTool):
    """Tool to resolve potentially fuzzy user-provided hierarchy names (like branches, libraries)
    against the exact names stored in the 'hierarcyCaches' table for the organization associated with the request context.
    It uses advanced matching to find the best match and returns the exact database name, ID, and matching score."""
    name: str = "hierarchy_name_resolver"
    description: str = (
        "Resolves user-provided hierarchy entity names (e.g., 'Main Library', 'Argyle') against the exact names "
        "in the database for the relevant organization. Use this *before* querying data if the user mentions "
        "specific branches, libraries, or other hierarchy entities by name. Returns a mapping of input names "
        "to their resolved database name, ID, and matching score."
    )
    args_schema: Type[BaseModel] = HierarchyResolverInput
    organization_id: str # Passed during instantiation for context, **used internally**
    min_score_threshold: int = 85
    db_name: str = "report_management" # Assume we always use this DB for hierarchy
    
    # Common words to handle specially in library names - marked as ClassVar to avoid Pydantic treating it as a field
    COMMON_WORDS: ClassVar[List[str]] = ["library", "branch", "center", "location", "regional", "public", "community", "the"]
    
    # Priority branch names that should have special handling - common/important branch identifiers
    PRIORITY_NAMES: ClassVar[List[str]] = ["main", "central", "downtown", "headquarters", "hq", "primary"]
    
    # Advanced name processing
    def _preprocess_name(self, name: str) -> Dict[str, Any]:
        """Preprocess a name with multiple approaches for more robust matching."""
        if not name:
            return {"original": "", "normalized": "", "keywords": [], "identifiers": []}
            
        # Basic normalization
        normalized = name.lower().strip()
        
        # Extract abbreviation if present in parentheses
        abbr_match = re.search(r'\(([A-Za-z]+)\)$', normalized)
        abbreviation = abbr_match.group(1).lower() if abbr_match else None
        
        # Remove abbreviation part for cleaner processing
        if abbr_match:
            normalized = normalized[:abbr_match.start()].strip()
        
        # Remove punctuation and special characters
        cleaned = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Tokenize
        tokens = [t.strip() for t in cleaned.split() if t.strip()]
        
        # Extract keywords (non-common words)
        keywords = [t for t in tokens if t.lower() not in self.COMMON_WORDS]
        
        # Check if we have priority keywords
        has_priority_name = any(kw.lower() in self.PRIORITY_NAMES for kw in keywords)
        
        # Create variant without common words for matching
        identifiers = " ".join(keywords)
        
        return {
            "original": name,
            "normalized": normalized,
            "keywords": keywords,
            "identifiers": identifiers,
            "abbreviation": abbreviation,
            "has_priority_name": has_priority_name,
            "priority_match": next((kw for kw in keywords if kw.lower() in self.PRIORITY_NAMES), None)
        }

    def _score_match(self, candidate_processed: Dict[str, Any], db_entry_processed: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate a comprehensive match score using multiple metrics."""
        scores = {}
        result = {"match": False, "score": 0, "method": None}
        
        # 1. Try exact match on normalized name
        if candidate_processed["normalized"] == db_entry_processed["normalized"]:
            result = {"match": True, "score": 100, "method": "exact"}
            return result
        
        # 2. Check for abbreviation match
        if (candidate_processed["abbreviation"] and db_entry_processed["abbreviation"] and
            candidate_processed["abbreviation"] == db_entry_processed["abbreviation"]):
            result = {"match": True, "score": 95, "method": "abbreviation"}
            return result
        
        # 3. Check for keyword exact match (if they have the same non-common words)
        if (candidate_processed["keywords"] and db_entry_processed["keywords"] and
            set(candidate_processed["keywords"]) == set(db_entry_processed["keywords"])):
            result = {"match": True, "score": 90, "method": "keywords"}
            return result
        
        # 4. Special priority name handling (e.g., "Main" should match "Main Library" very strongly)
        if candidate_processed["has_priority_name"] and db_entry_processed["has_priority_name"]:
            priority_name_candidate = candidate_processed["priority_match"]
            priority_name_db = db_entry_processed["priority_match"]
            
            if priority_name_candidate and priority_name_db and priority_name_candidate.lower() == priority_name_db.lower():
                result = {"match": True, "score": 95, "method": "priority_name"}
                return result
        
        # 5. If candidate has a priority name, check if it exists in the DB entry keywords
        if candidate_processed["has_priority_name"] and candidate_processed["priority_match"]:
            priority_name = candidate_processed["priority_match"].lower()
            db_keywords_lower = [k.lower() for k in db_entry_processed["keywords"]]
            
            if priority_name in db_keywords_lower:
                # This is a strong signal for matching, e.g., "Main Branch" should match "Main Library"
                priority_match_score = 92
                result = {"match": True, "score": priority_match_score, "method": "priority_in_db"}
                return result
        
        # 6. Check for leading keyword match (important for branch names like "Main Library" matching "Main")
        if (candidate_processed["keywords"] and db_entry_processed["keywords"] and
            len(candidate_processed["keywords"]) > 0 and len(db_entry_processed["keywords"]) > 0):
            
            candidate_first = candidate_processed["keywords"][0].lower()
            db_first = db_entry_processed["keywords"][0].lower()
            
            # Special case for full exact match on first keyword
            if candidate_first == db_first:
                # Check if the word is a priority name for higher score
                if candidate_first in self.PRIORITY_NAMES:
                    result = {"match": True, "score": 92, "method": "primary_priority_keyword"}
                else:
                    result = {"match": True, "score": 87, "method": "primary_keyword"}
                return result
            
            # Check if first character matches and one is a prefix of the other
            # This is to prevent "Main" matching with "Mandarin" too strongly
            if candidate_first[0] == db_first[0]:
                min_len = min(len(candidate_first), len(db_first))
                prefix_match_len = 0
                
                for i in range(min_len):
                    if candidate_first[i] == db_first[i]:
                        prefix_match_len += 1
                    else:
                        break
                
                # Calculate percentage of the shorter word that matches as prefix
                prefix_match_percent = prefix_match_len / min_len
                
                # Only consider a good partial prefix match if > 80% of the shorter word matches
                if prefix_match_percent > 0.8 and prefix_match_len >= 3:
                    # But reduce score for longer words that only share a prefix
                    partial_prefix_score = 72 + (prefix_match_percent * 10) # 72-82 range
                    
                    # Check if short word "main" is being matched against longer word like "mandarin"
                    # Stronger words like "main" should have lower prefix match score with longer words
                    if candidate_first in self.PRIORITY_NAMES and len(db_first) > len(candidate_first) + 3:
                        partial_prefix_score -= 15  # Significant penalty
                    
                    if db_first in self.PRIORITY_NAMES and len(candidate_first) > len(db_first) + 3:
                        partial_prefix_score -= 15  # Significant penalty
                        
                    result = {
                        "match": partial_prefix_score >= self.min_score_threshold,
                        "score": partial_prefix_score,
                        "method": "partial_prefix"
                    }
                    if result["match"]:
                        return result
                        
        # 7. Check for significant keyword containment (e.g., "Main" in "Main Library")
        candidate_kw_set = set(candidate_processed["keywords"])
        db_kw_set = set(db_entry_processed["keywords"])
        
        if candidate_kw_set and db_kw_set:
            # If ALL candidate keywords are in the DB entry
            if candidate_kw_set.issubset(db_kw_set):
                containment_score = 82
                result = {"match": True, "score": containment_score, "method": "keyword_containment"}
                return result
                
            # If multiple keywords match but not all
            intersection = candidate_kw_set.intersection(db_kw_set)
            if len(intersection) > 0:
                # Score based on percentage of matching keywords
                containment_ratio = len(intersection) / max(len(candidate_kw_set), len(db_kw_set))
                keyword_match_score = int(75 * containment_ratio)
                if keyword_match_score >= self.min_score_threshold:
                    result = {"match": True, "score": keyword_match_score, "method": "partial_keywords"}
                    return result
        
        # 8. Fallback to advanced string similarity metrics for fuzzy matching
        # Token sort ratio - less affected by word order
        token_sort_ratio = fuzz.token_sort_ratio(
            candidate_processed["normalized"], 
            db_entry_processed["normalized"]
        )
        scores["token_sort"] = token_sort_ratio
        
        # Partial ratio helps with substring matches
        partial_ratio = fuzz.partial_ratio(
            candidate_processed["normalized"], 
            db_entry_processed["normalized"]
        )
        
        # Apply a penalty for partial ratio when dealing with priority names
        # to prevent "Main" from matching with "Mandarin" just because they both start with "Ma"
        if (candidate_processed["has_priority_name"] or db_entry_processed["has_priority_name"]):
            # Check if the DB name is significantly longer than priority name
            if candidate_processed["has_priority_name"] and candidate_processed["priority_match"]:
                priority_len = len(candidate_processed["priority_match"])
                normalized_len = len(db_entry_processed["normalized"])
                
                # Stronger penalty for short priority names matching with much longer strings
                if normalized_len > priority_len * 1.5:
                    partial_ratio = max(50, partial_ratio - 15)  # Penalize partial matches for priority names
        
        scores["partial"] = partial_ratio
        
        # Token set ratio - good for overlapping words regardless of order
        token_set_ratio = fuzz.token_set_ratio(
            candidate_processed["normalized"], 
            db_entry_processed["normalized"]
        )
        scores["token_set"] = token_set_ratio
        
        # Maximum score across all fuzzy metrics
        max_fuzzy_score = max(scores.values()) if scores else 0
        
        # For fuzzy matches, use a higher threshold if we're dealing with priority names
        fuzzy_threshold = self.min_score_threshold
        if candidate_processed["has_priority_name"] or db_entry_processed["has_priority_name"]:
            fuzzy_threshold = max(self.min_score_threshold + 5, 90)  # More strict for priority names
        
        # Only accept if above threshold
        if max_fuzzy_score >= fuzzy_threshold:
            best_method = max(scores.items(), key=lambda x: x[1])[0]
            result = {
                "match": True,
                "score": max_fuzzy_score,
                "method": f"fuzzy_{best_method}"
            }
        
        return result

    async def _run(self, name_candidates: List[str], **kwargs: Any) -> Dict[str, Any]:
        org_id_to_use = self.organization_id
        log_prefix = f"[Org: {org_id_to_use}] [HierarchyResolver] " # Added prefix
        logger.info(f"{log_prefix}Executing with {len(name_candidates)} candidates: {name_candidates[:5]}..." ) # Log start, limit candidates shown

        try:
            async with get_async_db_connection(self.db_name) as conn:
                results = await self._execute_logic(conn, name_candidates, org_id_to_use)
                found_count = sum(1 for r in results.get("resolution_results", {}).values() if r["status"] == "found")
                logger.info(f"{log_prefix}Completed. Found {found_count}/{len(name_candidates)} candidates.") # Log completion summary
                return results
        except Exception as e:
            error_msg = f"Failed during async execution: {str(e)}"
            logger.error(f"{log_prefix}Error: {error_msg}", exc_info=True) # Log error with prefix
            # Propagate error in a structured way if possible, or raise
            return {"resolution_results": {name: {"status": "error", "message": f"Tool error: {str(e)}"} for name in name_candidates}}
            # raise ValueError(error_msg) # Or re-raise if preferred

    async def _execute_logic(self, connection: AsyncConnection, name_candidates: List[str], organization_id: str) -> Dict[str, Any]:
        log_prefix = f"[Org: {organization_id}] [HierarchyResolver] " # Add prefix here too for consistency
        resolved_map: Dict[str, Dict[str, Any]] = {}
        hierarchy_cache_data: List[Dict[str, Any]] = []

        # 1. Fetch hierarchy names belonging ONLY to the specified organization OR its direct children
        try:
            # Select ID and Name for non-deleted entries where the ID is the org ID itself,
            # OR the parentId is the org ID (i.e., direct children/locations).
            query = sqlalchemy.text(
                'SELECT "id", "name", "shortName" FROM "hierarchyCaches" ' \
                'WHERE "deletedAt" IS NULL AND (id = :org_id OR "parentId" = :org_id)'
            )
            # Pass the organization_id as a parameter
            result = await connection.execute(query, {"org_id": organization_id})
            # Process the result mapping without awaiting
            mappings_result = result.mappings()
            hierarchy_cache_data = [
                {
                    "id": str(row.id), 
                    "name": row.name,
                    "shortName": row.shortName if hasattr(row, "shortName") else None
                } 
                for row in mappings_result
            ]
            
            logger.debug(f"{log_prefix}Retrieved {len(hierarchy_cache_data)} hierarchy entries.") # Keep DEBUG for data retrieval

            if not hierarchy_cache_data:
                 logger.warning(f"No hierarchy entries found for organization {organization_id} or its direct children. Cannot resolve names.")
                 for name in name_candidates:
                      resolved_map[name] = {"status": "no_hierarchy_data", "resolved_name": None, "id": None, "score": 0}
                 return {"resolution_results": resolved_map}

        except Exception as e:
            logger.error(f"{log_prefix}Database error fetching hierarchy cache: {e}", exc_info=True) # Log DB error
            db_error_msg = f"DB error fetching org/children cache: {str(e)}"
            for name in name_candidates:
                 resolved_map[name] = {"status": "error", "error_message": db_error_msg, "resolved_name": None, "id": None, "score": 0}
            return {"resolution_results": resolved_map, "error": f"Database error fetching org/children hierarchy data: {str(e)}"}

        # 2. Pre-process DB entries for better matching
        db_entries_processed = []
        for entry in hierarchy_cache_data:
            name_processed = self._preprocess_name(entry["name"])
            # Also process shortName if available
            shortname_processed = self._preprocess_name(entry["shortName"]) if entry["shortName"] else None
            
            db_entries_processed.append({
                "id": entry["id"],
                "name": entry["name"],
                "name_processed": name_processed,
                "shortname": entry["shortName"],
                "shortname_processed": shortname_processed
            })

        # 3. Process each candidate name
        for candidate in name_candidates:
            matched = False
            try:
                # Preprocess the candidate name
                candidate_processed = self._preprocess_name(candidate)
                
                # Score against all DB entries using our enhanced matching algorithm
                matches = []
                for db_entry in db_entries_processed:
                    # Try matching against full name first
                    match_result = self._score_match(candidate_processed, db_entry["name_processed"])
                    
                    # If shortname exists and no high-confidence match yet, also try matching against shortname
                    if db_entry["shortname_processed"] and (not match_result["match"] or match_result["score"] < 90):
                        shortname_match = self._score_match(candidate_processed, db_entry["shortname_processed"])
                        # Take the better of the two matches
                        if shortname_match["score"] > match_result["score"]:
                            match_result = shortname_match
                    
                    if match_result["match"]:
                        matches.append({
                            "db_entry": db_entry,
                            "score": match_result["score"],
                            "method": match_result["method"]
                        })
                
                # Find the best match if any
                if matches:
                    # Sort by score descending
                    matches.sort(key=lambda x: x["score"], reverse=True)
                    best_match = matches[0]
                    
                    matched_name = best_match["db_entry"]["name"]
                    matched_id = best_match["db_entry"]["id"]
                    match_score = best_match["score"]
                    match_method = best_match["method"]
                    
                    logger.debug(f"{log_prefix}Resolved '{candidate}' -> '{matched_name}' (Score: {match_score}, Method: {match_method})")
                    resolved_map[candidate] = {
                        "status": "found",
                        "resolved_name": matched_name,
                        "id": matched_id,
                        "score": match_score,
                        "method": match_method
                    }
                    matched = True
                    
                # If still no match
                if not matched:
                    logger.warning(f"{log_prefix}Could not resolve '{candidate}' (score >= {self.min_score_threshold})") # Keep WARN for failure
                    resolved_map[candidate] = {
                        "status": "not_found",
                        "resolved_name": None,
                        "id": None,
                        "score": 0
                    }

            except Exception as e:
                 logger.error(f"{log_prefix}Error during matching for '{candidate}': {e}", exc_info=True) # Log matching error
                 resolved_map[candidate] = {
                     "status": "error",
                     "error_message": f"Matching error: {str(e)}",
                     "resolved_name": None,
                     "id": None,
                     "score": 0
                 }

        logger.debug(f"Hierarchy name resolution completed. Result map: {resolved_map}")
        # Final successful return structure
        return {"resolution_results": resolved_map}
        
    # Implement ainvoke to ensure compatibility with BaseTool
    async def ainvoke(self, input_data: Dict[str, Any], **kwargs: Any) -> Any:
        """Override ainvoke to ensure it calls our async _run method."""
        validated_input = self.args_schema(**input_data)
        return await self._run(**validated_input.dict(), **kwargs)