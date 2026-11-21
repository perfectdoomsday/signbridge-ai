from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "backend/.env"), extra="ignore")

    app_name: str = "SignBridge AI"
    model_path: str = "artifacts/isl_mobilenet.onnx"
    sign_language: str = "ISL"
    confidence_threshold: float = 0.60
    image_size: int = 224
    max_upload_mb: int = 10
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


settings = Settings()
