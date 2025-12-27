"""
Supabase ML Storage Module

Stores all AI/ML training data and models in Supabase cloud database:
- Training data (predictions & outcomes)
- Trained model metadata and files
- Model performance metrics
- Training statistics

Author: Claude AI Assistant
Date: 2025-12-27
"""

import os
import json
import pickle
import base64
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingRecord:
    """Single training data record"""
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    symbol: str = ""
    prediction: str = ""  # BUY, SELL, HOLD
    confidence: float = 0.0
    features: Dict = None
    actual_outcome: Optional[str] = None  # CORRECT, INCORRECT, PENDING
    outcome_timestamp: Optional[datetime] = None
    profit_loss: Optional[float] = None
    notes: str = ""


@dataclass
class ModelMetadata:
    """Trained model metadata"""
    id: Optional[int] = None
    model_name: str = ""
    version: int = 1
    created_at: Optional[datetime] = None
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    training_samples: int = 0
    feature_count: int = 0
    hyperparameters: Dict = None
    model_file_url: Optional[str] = None  # Supabase Storage URL
    is_active: bool = False
    notes: str = ""


class SupabaseMLStorage:
    """Manages ML data storage in Supabase"""

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """
        Initialize Supabase ML Storage

        Args:
            supabase_url: Supabase project URL (from env if not provided)
            supabase_key: Supabase anon key (from env if not provided)
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase package not installed. Run: pip install supabase")

        # Get credentials from environment or parameters
        self.url = supabase_url or os.environ.get('SUPABASE_URL')
        self.key = supabase_key or os.environ.get('SUPABASE_KEY')

        if not self.url or not self.key:
            raise ValueError("Supabase URL and KEY required. Set SUPABASE_URL and SUPABASE_KEY environment variables.")

        # Initialize client
        try:
            self.client: Client = create_client(self.url, self.key)
            logger.info("✅ Supabase ML Storage initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase: {e}")
            raise

        # Table names
        self.TRAINING_TABLE = 'ml_training_data'
        self.MODELS_TABLE = 'ml_models'
        self.STATS_TABLE = 'ml_statistics'

    # ========================================================================
    # TRAINING DATA OPERATIONS
    # ========================================================================

    def save_training_record(self, record: TrainingRecord) -> bool:
        """
        Save a single training record to Supabase

        Args:
            record: TrainingRecord object

        Returns:
            bool: Success status
        """
        try:
            # Convert to dict
            data = asdict(record)

            # Remove id if None (let Supabase auto-generate)
            if data.get('id') is None:
                data.pop('id', None)

            # Convert datetime to ISO string
            if data.get('timestamp'):
                data['timestamp'] = data['timestamp'].isoformat()
            if data.get('outcome_timestamp'):
                data['outcome_timestamp'] = data['outcome_timestamp'].isoformat()

            # Convert features dict to JSON string
            if data.get('features'):
                data['features'] = json.dumps(data['features'])

            # Insert into Supabase
            result = self.client.table(self.TRAINING_TABLE).insert(data).execute()

            logger.info(f"✅ Saved training record: {record.symbol} - {record.prediction}")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving training record: {e}")
            return False

    def get_training_data(
        self,
        limit: int = 1000,
        symbol: Optional[str] = None,
        outcome_only: bool = False
    ) -> List[TrainingRecord]:
        """
        Retrieve training data from Supabase

        Args:
            limit: Maximum records to retrieve
            symbol: Filter by symbol (optional)
            outcome_only: Only get records with outcomes

        Returns:
            List of TrainingRecord objects
        """
        try:
            # Build query
            query = self.client.table(self.TRAINING_TABLE).select("*")

            # Apply filters
            if symbol:
                query = query.eq('symbol', symbol)

            if outcome_only:
                query = query.not_.is_('actual_outcome', 'null')

            # Execute query
            query = query.limit(limit).order('timestamp', desc=True)
            result = query.execute()

            # Convert to TrainingRecord objects
            records = []
            for row in result.data:
                # Parse features JSON
                if row.get('features'):
                    row['features'] = json.loads(row['features'])

                # Parse timestamps
                if row.get('timestamp'):
                    row['timestamp'] = datetime.fromisoformat(row['timestamp'])
                if row.get('outcome_timestamp'):
                    row['outcome_timestamp'] = datetime.fromisoformat(row['outcome_timestamp'])

                records.append(TrainingRecord(**row))

            logger.info(f"✅ Retrieved {len(records)} training records")
            return records

        except Exception as e:
            logger.error(f"❌ Error retrieving training data: {e}")
            return []

    def update_outcome(
        self,
        record_id: int,
        actual_outcome: str,
        profit_loss: Optional[float] = None
    ) -> bool:
        """
        Update the outcome of a training record

        Args:
            record_id: Record ID
            actual_outcome: CORRECT, INCORRECT, etc.
            profit_loss: Profit/loss amount (optional)

        Returns:
            bool: Success status
        """
        try:
            update_data = {
                'actual_outcome': actual_outcome,
                'outcome_timestamp': datetime.now().isoformat()
            }

            if profit_loss is not None:
                update_data['profit_loss'] = profit_loss

            result = self.client.table(self.TRAINING_TABLE)\
                .update(update_data)\
                .eq('id', record_id)\
                .execute()

            logger.info(f"✅ Updated outcome for record {record_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error updating outcome: {e}")
            return False

    # ========================================================================
    # MODEL OPERATIONS
    # ========================================================================

    def save_model_metadata(self, metadata: ModelMetadata) -> Optional[int]:
        """
        Save model metadata to Supabase

        Args:
            metadata: ModelMetadata object

        Returns:
            int: Model ID if successful, None otherwise
        """
        try:
            data = asdict(metadata)

            # Remove id if None
            if data.get('id') is None:
                data.pop('id', None)

            # Convert datetime
            if data.get('created_at'):
                data['created_at'] = data['created_at'].isoformat()

            # Convert hyperparameters to JSON
            if data.get('hyperparameters'):
                data['hyperparameters'] = json.dumps(data['hyperparameters'])

            # Insert
            result = self.client.table(self.MODELS_TABLE).insert(data).execute()

            model_id = result.data[0]['id'] if result.data else None
            logger.info(f"✅ Saved model metadata: {metadata.model_name} v{metadata.version} (ID: {model_id})")
            return model_id

        except Exception as e:
            logger.error(f"❌ Error saving model metadata: {e}")
            return None

    def upload_model_file(self, model_id: int, model_object, model_name: str) -> Optional[str]:
        """
        Upload trained model file to Supabase Storage

        Args:
            model_id: Model metadata ID
            model_object: Trained model object (will be pickled)
            model_name: Model name for filename

        Returns:
            str: Public URL of uploaded model, None if failed
        """
        try:
            # Pickle the model
            model_bytes = pickle.dumps(model_object)

            # Encode to base64 for storage
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')

            # Create filename
            filename = f"models/{model_name}_v{model_id}.pkl"

            # Upload to Supabase Storage
            # Note: You need to create a 'models' bucket in Supabase first
            result = self.client.storage.from_('models').upload(
                filename,
                model_bytes,
                file_options={"content-type": "application/octet-stream"}
            )

            # Get public URL
            public_url = self.client.storage.from_('models').get_public_url(filename)

            # Update model metadata with URL
            self.client.table(self.MODELS_TABLE)\
                .update({'model_file_url': public_url})\
                .eq('id', model_id)\
                .execute()

            logger.info(f"✅ Uploaded model file: {filename}")
            return public_url

        except Exception as e:
            logger.error(f"❌ Error uploading model file: {e}")
            return None

    def download_model(self, model_id: int):
        """
        Download and deserialize a trained model

        Args:
            model_id: Model metadata ID

        Returns:
            Trained model object, or None if failed
        """
        try:
            # Get model metadata
            result = self.client.table(self.MODELS_TABLE)\
                .select('model_file_url, model_name')\
                .eq('id', model_id)\
                .execute()

            if not result.data:
                logger.error(f"Model {model_id} not found")
                return None

            model_url = result.data[0]['model_file_url']

            if not model_url:
                logger.error(f"Model {model_id} has no file URL")
                return None

            # Extract filename from URL
            filename = model_url.split('/')[-1]

            # Download from Supabase Storage
            file_data = self.client.storage.from_('models').download(f"models/{filename}")

            # Deserialize
            model_object = pickle.loads(file_data)

            logger.info(f"✅ Downloaded model {model_id}")
            return model_object

        except Exception as e:
            logger.error(f"❌ Error downloading model: {e}")
            return None

    def get_active_model(self) -> Optional[Tuple[int, any]]:
        """
        Get the currently active model

        Returns:
            Tuple of (model_id, model_object) or (None, None)
        """
        try:
            # Query for active model
            result = self.client.table(self.MODELS_TABLE)\
                .select('*')\
                .eq('is_active', True)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()

            if not result.data:
                logger.warning("No active model found")
                return None, None

            model_meta = result.data[0]
            model_id = model_meta['id']

            # Download model
            model_object = self.download_model(model_id)

            return model_id, model_object

        except Exception as e:
            logger.error(f"❌ Error getting active model: {e}")
            return None, None

    def set_active_model(self, model_id: int) -> bool:
        """
        Set a model as the active model (deactivate all others)

        Args:
            model_id: Model ID to activate

        Returns:
            bool: Success status
        """
        try:
            # Deactivate all models
            self.client.table(self.MODELS_TABLE)\
                .update({'is_active': False})\
                .neq('id', 0)\
                .execute()

            # Activate selected model
            self.client.table(self.MODELS_TABLE)\
                .update({'is_active': True})\
                .eq('id', model_id)\
                .execute()

            logger.info(f"✅ Set model {model_id} as active")
            return True

        except Exception as e:
            logger.error(f"❌ Error setting active model: {e}")
            return False

    # ========================================================================
    # STATISTICS OPERATIONS
    # ========================================================================

    def save_training_stats(self, stats: Dict) -> bool:
        """
        Save training statistics snapshot

        Args:
            stats: Statistics dictionary

        Returns:
            bool: Success status
        """
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'stats': json.dumps(stats)
            }

            self.client.table(self.STATS_TABLE).insert(data).execute()

            logger.info("✅ Saved training statistics")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving statistics: {e}")
            return False

    def get_latest_stats(self) -> Optional[Dict]:
        """
        Get the most recent training statistics

        Returns:
            Dict: Statistics or None
        """
        try:
            result = self.client.table(self.STATS_TABLE)\
                .select('stats')\
                .order('timestamp', desc=True)\
                .limit(1)\
                .execute()

            if not result.data:
                return None

            return json.loads(result.data[0]['stats'])

        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return None

    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    def initialize_tables(self) -> bool:
        """
        Initialize Supabase tables (run this once to set up schema)

        Note: You should create these tables in Supabase dashboard instead:

        CREATE TABLE ml_training_data (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            symbol TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence FLOAT,
            features JSONB,
            actual_outcome TEXT,
            outcome_timestamp TIMESTAMPTZ,
            profit_loss FLOAT,
            notes TEXT
        );

        CREATE TABLE ml_models (
            id BIGSERIAL PRIMARY KEY,
            model_name TEXT NOT NULL,
            version INT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            accuracy FLOAT,
            precision FLOAT,
            recall FLOAT,
            f1_score FLOAT,
            training_samples INT,
            feature_count INT,
            hyperparameters JSONB,
            model_file_url TEXT,
            is_active BOOLEAN DEFAULT FALSE,
            notes TEXT
        );

        CREATE TABLE ml_statistics (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            stats JSONB NOT NULL
        );

        Returns:
            bool: Instructions displayed
        """
        logger.info("""
        ⚠️ Please create tables in Supabase dashboard manually.
        See docstring of initialize_tables() for SQL commands.
        """)
        return True

    def get_stats_summary(self) -> Dict:
        """Get summary statistics from Supabase"""
        try:
            # Count total records
            total = self.client.table(self.TRAINING_TABLE).select('id', count='exact').execute()
            total_records = total.count if hasattr(total, 'count') else 0

            # Count with outcomes
            with_outcome = self.client.table(self.TRAINING_TABLE)\
                .select('id', count='exact')\
                .not_.is_('actual_outcome', 'null')\
                .execute()
            outcome_count = with_outcome.count if hasattr(with_outcome, 'count') else 0

            # Count models
            models = self.client.table(self.MODELS_TABLE).select('id', count='exact').execute()
            model_count = models.count if hasattr(models, 'count') else 0

            return {
                'total_records': total_records,
                'records_with_outcome': outcome_count,
                'pending_outcomes': total_records - outcome_count,
                'total_models': model_count
            }

        except Exception as e:
            logger.error(f"❌ Error getting stats summary: {e}")
            return {
                'total_records': 0,
                'records_with_outcome': 0,
                'pending_outcomes': 0,
                'total_models': 0
            }
