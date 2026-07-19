use crate::llm::Message;
use std::fmt::Debug;

pub trait Memory: Debug + Send + Sync {
    fn window(&self, history: &[Message]) -> Vec<Message>;
}

pub mod sliding;

pub use sliding::SlidingWindow;
