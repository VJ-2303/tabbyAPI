"""Contains stats types."""

from typing import Optional

from pydantic import BaseModel, Field


class StatsServer(BaseModel):
    """Identity of the running server and its statistics document."""

    version: int = Field(0, description="Version of the statistics document format")
    now: float = Field(0.0, description="Server time this snapshot was taken, Unix seconds")
    started_at: Optional[float] = Field(
        None, description="When statistics were first recorded for this installation"
    )
    sessions: int = Field(0, description="Server runs that have written to the document")


class StatsGpu(BaseModel):
    """Memory of one GPU, when it can be read."""

    index: int = Field(0, description="Device index")
    name: str = Field("", description="Device name")
    used_bytes: int = Field(0, description="Memory in use")
    total_bytes: int = Field(0, description="Total memory")
    free_bytes: int = Field(0, description="Memory free")


class StatsModel(BaseModel):
    """Parameters of the currently loaded model."""

    id: Optional[str] = Field(None, description="Model directory name")
    path: Optional[str] = Field(None, description="Full path of the model directory")
    max_seq_len: Optional[int] = Field(None, description="Loaded context length")
    cache_mode: Optional[str] = Field(None, description="Quantization of the KV cache")
    cache_size: Optional[int] = Field(None, description="KV cache size in tokens")
    max_batch_size: Optional[int] = Field(None, description="Maximum concurrent sequences")
    chunk_size: Optional[int] = Field(None, description="Prefill chunk size")
    use_vision: Optional[bool] = Field(None, description="Whether the model takes images")
    prompt_template: Optional[str] = Field(None, description="Name of the chat template")
    n_ctx_train: Optional[int] = Field(
        None, description="Context length the model was trained with"
    )
    n_vocab: Optional[int] = Field(None, description="Vocabulary size")
    n_embd: Optional[int] = Field(None, description="Embedding dimension")
    size_bytes: Optional[int] = Field(None, description="Size of the weight files")


class StatsCache(BaseModel):
    """
    KV cache usage, as the backend reports it. Values a backend doesn't provide
    are null, and a key it doesn't know is not part of this response.
    """

    max_tokens: Optional[int] = Field(None, description="Cache capacity in tokens")
    used_tokens: Optional[int] = Field(None, description="Tokens held by in-flight jobs")
    cached_tokens: Optional[int] = Field(None, description="Tokens reusable from a prior prompt")
    tier_max_tokens: Optional[int] = Field(None, description="System memory tier capacity")
    tier_cached_tokens: Optional[int] = Field(None, description="Tokens held in system memory")
    disk_max_bytes: Optional[int] = Field(None, description="Disk cache capacity in bytes")
    disk_used_bytes: Optional[int] = Field(None, description="Disk cache used in bytes")
    disk_cached_tokens: Optional[int] = Field(None, description="Tokens held on disk")
    hit_rate: Optional[float] = Field(None, description="Share of prompt tokens found in cache")
    tier_hit_rate: Optional[float] = Field(None, description="Share found in system memory or disk")
    disk_hit_rate: Optional[float] = Field(None, description="Share found on disk")
    pending_jobs: Optional[int] = Field(None, description="Jobs queued in the engine")


class StatsJob(BaseModel):
    """One in-flight generation."""

    label: Optional[str] = Field(None, description="Console label of the request")
    stage: str = Field("queued", description="queued, prefill or generating")
    prompt_tokens: int = Field(0, description="Prompt tokens for this job")
    prefill_tokens: int = Field(0, description="Prompt tokens read so far")
    cached_tokens: int = Field(0, description="Prompt tokens served from cache")
    gen_tokens: int = Field(0, description="Tokens generated so far")
    gen_tokens_per_sec: Optional[float] = Field(
        None, description="Generation rate over the last few seconds"
    )


class StatsLive(BaseModel):
    """What the server is doing right now."""

    jobs: list[StatsJob] = Field(default_factory=list, description="In-flight generations")
    active: int = Field(0, description="Jobs currently being generated")
    queued: Optional[int] = Field(None, description="Jobs waiting in the engine")
    generating_tokens_per_sec: Optional[float] = Field(
        None, description="Combined rate across in-flight generations"
    )
    cache: Optional[StatsCache] = Field(None, description="KV cache usage")


class StatsCounters(BaseModel):
    """Everything counted for one scope, with rates derived from the totals."""

    requests: int = Field(0, description="Generation requests")
    gen_tokens: int = Field(0, description="Tokens generated")
    prompt_tokens: int = Field(0, description="Prompt tokens, cached ones included")
    processed_tokens: int = Field(0, description="Prompt tokens that had to be read")
    cached_tokens: int = Field(0, description="Prompt tokens served from cache")
    draft_accept: int = Field(0, description="Draft tokens accepted")
    draft_reject: int = Field(0, description="Draft tokens rejected")
    failed: int = Field(0, description="Requests that ended in an error")
    cancelled: int = Field(0, description="Requests cancelled by the client")
    embedding_requests: int = Field(0, description="Embeddings requests")
    embedding_tokens: int = Field(0, description="Tokens embedded")
    prompt_time: float = Field(0.0, description="Seconds spent reading prompts")
    gen_time: float = Field(0.0, description="Seconds spent generating")
    queue_time: float = Field(0.0, description="Seconds spent queued")
    total_time: float = Field(0.0, description="Seconds from request to finish")
    embedding_time: float = Field(0.0, description="Seconds spent embedding")
    finish: dict[str, int] = Field(default_factory=dict, description="Counts per finish reason")
    gen_tokens_per_sec: Optional[float] = Field(
        None, description="Tokens generated per second over all generations"
    )
    prompt_tokens_per_sec: Optional[float] = Field(
        None, description="Prompt tokens processed per second over all requests"
    )
    gen_tokens_per_sec_p50: Optional[float] = Field(
        None, description="Median generation rate of the requests seen this session"
    )
    gen_tokens_per_sec_p95: Optional[float] = Field(
        None, description="95th percentile generation rate of this session"
    )
    uptime: Optional[int] = Field(None, description="Seconds since this server run started")


class StatsModelCounters(StatsCounters):
    """Counters for one model, over every session."""

    name: str = Field("", description="Model directory name")
    loads: int = Field(0, description="Times this model was loaded")
    last_loaded: float = Field(0.0, description="When this model was last loaded, Unix seconds")


class StatsBucket(BaseModel):
    """One hour or day of history."""

    hour: Optional[int] = Field(None, description="Start of the hour, Unix seconds")
    day: Optional[int] = Field(None, description="Start of the day, Unix seconds")
    requests: int = Field(0, description="Generation requests")
    gen_tokens: int = Field(0, description="Tokens generated")
    prompt_tokens: int = Field(0, description="Prompt tokens")
    processed_tokens: int = Field(0, description="Prompt tokens that had to be read")
    cached_tokens: int = Field(0, description="Prompt tokens served from cache")
    embedding_requests: int = Field(0, description="Embeddings requests")
    embedding_tokens: int = Field(0, description="Tokens embedded")


class StatsHistory(BaseModel):
    """Recent counters as hourly and daily buckets, oldest first."""

    hourly: list[StatsBucket] = Field(default_factory=list, description="Recent hours")
    daily: list[StatsBucket] = Field(default_factory=list, description="Recent days")


class StatsIncident(BaseModel):
    """An error that made the service unhealthy."""

    time: float = Field(0.0, description="When the error occurred, Unix seconds")
    description: str = Field("", description="The error message")


class StatsResponse(BaseModel):
    """Live and historical statistics of the server."""

    server: StatsServer = Field(default_factory=StatsServer)
    model: Optional[StatsModel] = Field(None, description="Loaded model, null when there is none")
    gpu: list[StatsGpu] = Field(default_factory=list, description="GPUs the server can see")
    live: Optional[StatsLive] = Field(None, description="Current activity, null with no model")
    session: StatsCounters = Field(description="Counters since this server run started")
    lifetime: StatsCounters = Field(description="Counters over every session")
    models: list[StatsModelCounters] = Field(
        default_factory=list, description="Counters per model, most generated tokens first"
    )
    history: StatsHistory = Field(default_factory=StatsHistory)
    incidents: list[StatsIncident] = Field(default_factory=list, description="Recent errors")