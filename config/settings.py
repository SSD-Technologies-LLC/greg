from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Required
    telegram_bot_token: str
    anthropic_api_key: str
    greg_bot_username: str

    # Postgres — accepts DATABASE_URL directly (Railway) or builds from parts (Docker Compose)
    database_url: str | None = None
    postgres_user: str = "greg"
    postgres_password: str = ""
    postgres_db: str = "greg_brain"

    # Redis — accepts REDIS_URL directly (Railway) or builds from parts (Docker Compose)
    redis_url: str | None = None
    redis_password: str = ""

    # Tuning
    greg_response_threshold: float = 0.25
    greg_random_factor: float = 0.15
    greg_cooldown_messages: int = 3
    greg_max_unprompted_per_hour: int = 5
    greg_max_api_calls_per_hour: int = 60
    greg_max_response_tokens: int = 300
    greg_night_start: int = 1
    greg_night_end: int = 8
    greg_timezone: str = "Europe/Moscow"
    greg_distill_every_n: int = 50
    greg_distill_every_minutes: int = 30
    greg_redis_buffer_size: int = 200
    greg_health_port: int = 8080
    greg_log_level: str = "INFO"

    @property
    def postgres_dsn(self) -> str:
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@postgres:5432/{self.postgres_db}"

    @property
    def redis_dsn(self) -> str:
        if self.redis_url:
            return self.redis_url
        return f"redis://:{self.redis_password}@redis:6379/0"

    model_config = {"env_file": ".env"}


settings = Settings()
