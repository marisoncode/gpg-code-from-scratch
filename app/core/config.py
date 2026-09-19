from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Environment mode: "production" | "development"
    environment: str = "production"

    # Permissions & Facility APIs (Endpoint-based Identity & Access Resolution)
    # Required in production; empty by default to prevent silent misconfiguration.
    permissions_api: str = ""
    facility_api: str = ""
    production_api: str = ""
    compliance_api: str = ""
    order_api: str = ""
    notification_hub_api: str = ""
    collection_id: str = ""
    default_collection_id: str = "sales"

    # MongoDB / Cosmos DB (Mongo API) persistence
    # Required in production; empty by default to prevent silent misconfiguration.
    mongo_connection_string: str = ""
    mongo_database_name: str = ""
    mongo_threads_collection: str = "chat_threads"
    mongo_audit_collection: str = "audit_log"

    # Frontend origin for CORS
    # Required in production; empty by default to prevent silent misconfiguration.
    frontend_origin: str = ""
    cors_origins: str = "http://127.0.0.1:4200,http://localhost:4200"

    # AI configuration (strictly OpenAI-compatible)
    ai_provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = ""
    ai_model: str = "gpt-4o-mini"
    # Optional Azure OpenAI endpoint, e.g. https://YOUR-RESOURCE.openai.azure.com
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-08-01-preview"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        origins: set[str] = set()
        if self.frontend_origin:
            origins.add(self.frontend_origin.strip())
        for o in self.cors_origins.split(","):
            if o.strip():
                origins.add(o.strip())
        return sorted(origins)

    @property
    def is_production(self) -> bool:
        return (self.environment or "").strip().lower() == "production"

    @property
    def provider(self) -> str:
        return (self.ai_provider or "openai").strip().lower()


settings = Settings()

