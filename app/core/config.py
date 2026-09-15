from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Environment mode: "production" | "development"
    environment: str = "production"

    # Facility JWT auth
    jwt_secret: str = ""
    jwt_algorithms: str = "HS256"

    # Permissions API endpoint
    permissions_api: str = ""

    # Cosmos DB configuration
    # COSMOS_DB_READONLY_KEY: scoped strictly for business-data containers
    # COSMOS_DB_WRITE_KEY: scoped strictly for audit and chat-history containers
    cosmos_db_endpoint: str = ""
    cosmos_db_readonly_key: str = ""
    cosmos_db_write_key: str = ""
    cosmos_db_database: str = "cpg_compliance"
    cosmos_db_audit_container: str = "audit_trail"
    cosmos_db_chat_container: str = "chat_history"

    # Frontend origin for CORS
    frontend_origin: str = "http://localhost:4200"
    cors_origins: str = "http://127.0.0.1:4200,http://localhost:4200"

    # Provider: "openai" | "gemini" | "ollama"
    ai_provider: str = "openai"
    # Gemini key (only required when AI_PROVIDER=gemini)
    ai_api_key: str = ""
    # OpenAI / Azure OpenAI key
    openai_api_key: str = ""
    # e.g. gpt-4o-mini, gemini-2.0-flash, llama3.2
    ai_model: str = "gpt-4o-mini"
    # Optional Azure OpenAI endpoint, e.g. https://YOUR-RESOURCE.openai.azure.com
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-08-01-preview"
    # Local Ollama (AI_PROVIDER=ollama) — no API key required
    ollama_base_url: str = "http://127.0.0.1:11434"

    # Facility APIs — JWT-proxied dashboard reads
    facility_api: str = "https://facility-user-api.cpguardian.com/api"
    production_api: str = "https://production-api.cpguardian.com/api"
    compliance_api: str = "https://compliance-api.cpguardian.com/api"
    collection_id: str = "sales"

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

