use tracing_subscriber::{ fmt, EnvFilter};

#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    #[error("Invalid Log Filter Active: {0}")]
    InvalidFilter(#[from] tracing_subscriber::filter::ParseError),
    #[error("Traicing subscriber already set: {0}")]
    AlreadyInitialized(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),
}

pub fn build_filter(default_directive: &str) -> Result<EnvFilter, TelemetryError> {
    match EnvFilter::try_from_default_env() {
        Ok(filter) => Ok(filter),
        Err(_) => Ok(EnvFilter::try_new(default_directive)?)
    }
    // Ok(EnvFilter::try_new(default_directive)?)
}

// Install the global tracing subscriber. Call once, as early as possible.
pub fn init(default_directive: &str) -> Result<(), TelemetryError> {
    let filter = build_filter(default_directive)?;

    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(|| anstream::stdout())
        .try_init()?;
    Ok(())
}