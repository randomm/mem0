from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class PGVectorConfig(BaseModel):
    """
    Configuration for the PGVector vector store.

    Attributes:
        dbname: Name of the PostgreSQL database.
        collection_name: Name of the table used as the collection.
        embedding_model_dims: Dimension of the embedding vectors.
        user: Database username.
        password: Database password.
        host: Database host address.
        port: Database port number.
        diskann: Whether to use DiskANN index (requires pgvectorscale).
            Disabled if quantization is enabled.
        hnsw: Whether to use HNSW index.
        quantization: Optional dictionary to configure vector quantization.
            Currently supports `{"precision": "binary"}` or `{"precision": "ubinary"}`.
            Requires `sentence-transformers` library.
        matryoshka_dims: Optional target dimensions for Matryoshka embedding truncation.
    """
    dbname: str = Field("postgres", description="Default name for the database")
    collection_name: str = Field("mem0", description="Default name for the collection")
    embedding_model_dims: Optional[int] = Field(1536, description="Dimensions of the embedding model")
    user: Optional[str] = Field(None, description="Database user")
    password: Optional[str] = Field(None, description="Database password")
    host: Optional[str] = Field(None, description="Database host. Default is localhost")
    port: Optional[int] = Field(None, description="Database port. Default is 1536")
    diskann: Optional[bool] = Field(True, description="Use diskann for approximate nearest neighbors search")
    hnsw: Optional[bool] = Field(False, description="Use hnsw for faster search")
    quantization: Optional[Dict[str, Any]] = Field(
        None,
        description="(Optional) Quantization configuration. E.g., {'precision': 'binary' or 'ubinary'}."
    )
    matryoshka_dims: Optional[int] = Field(
        None,
        description="(Optional) Target dimensions for Matryoshka embedding truncation. Must be less than embedding_model_dims."
    )

    @model_validator(mode="before")
    def check_auth_and_connection(cls, values):
        """Validate that essential connection details are provided."""
        user, password = values.get("user"), values.get("password")
        host, port = values.get("host"), values.get("port")
        if not user and not password:
            raise ValueError("Both 'user' and 'password' must be provided.")
        if not host and not port:
            raise ValueError("Both 'host' and 'port' must be provided.")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_quantization_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the structure and values of the quantization config."""
        quantization_config = values.get("quantization")
        if quantization_config:
            precision = quantization_config.get("precision")
            if not precision:
                raise ValueError("'precision' key is required within 'quantization' config.")
            if precision not in ["binary", "ubinary"]:
                raise ValueError(
                    f"Unsupported precision '{precision}'. Only 'binary' and 'ubinary' are currently supported for pgvector."
                )
            # Add more validation if other keys are added later (e.g., for int8 calibration)
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure no unexpected fields are passed in the config."""
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    @model_validator(mode='after')
    def validate_matryoshka_dims(self):
        """Validate that matryoshka_dims is less than embedding_model_dims if provided."""
        if (
            self.matryoshka_dims is not None and 
            self.embedding_model_dims is not None and 
            self.matryoshka_dims >= self.embedding_model_dims  # Use >= for validation
        ):
            raise ValueError(
                f"matryoshka_dims ({self.matryoshka_dims}) must be less than "
                f"embedding_model_dims ({self.embedding_model_dims})"
            )
        return self
