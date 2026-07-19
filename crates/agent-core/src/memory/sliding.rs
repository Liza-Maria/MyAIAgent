use super::Memory;
use crate::llm::{ Message, Role };

#[derive(Debug)]
pub struct SlidingWindow {
    pub max_messages: usize,
}

impl Memory for SlidingWindow {
    fn window(&self, history: &[Message]) -> Vec<Message> {
        let mut result = Vec::new();

        let keep = if history.first().map(|message| message.role) == Some(Role::System) {
            result.push(history[0].clone());
            &history[1..]
        } else {
            history
        };

        let start = keep.len().saturating_sub(self.max_messages);

        if start > 0 && matches!(
            keep.get(start).map(|message| &message.role),
            Some(Role::Tool)
        ) {
            let mut pos = start - 1;
            while pos >= 0 {
                if  matches!(
                    history.get(pos).map(|message| &message.role),
                    Some(Role::Assistant)
                ) {
                    result.push(history[pos].clone());
                    break;
                }
                
                pos -= 1;
            }
        }

        result.extend_from_slice(&keep[start..]);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_user_and_last_n_messages() {
        let history = vec![
            Message::system("system"),
            Message::user("user 1"),
            Message::assistant("assistant 1"),
            Message::user("user 2"),
            Message::assistant("assistant 2"),
            Message::user("user 3"),
            Message::assistant("assistant 3"),
            Message::user("user 4"),
            Message::assistant("assistant 4"),
            Message::user("user 5")
        ];

        let memory = SlidingWindow {
            max_messages: 4,
        };

        let result = memory.window(&history);

        assert_eq!(result.len(), 5);
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[0].content, Some("system".to_string()));
        assert_eq!(result[1].content, Some("assistant 3".to_string()));
        assert_eq!(result[2].content, Some("user 4".to_string()));
        assert_eq!(result[3].content, Some("assistant 4".to_string()));
        assert_eq!(result[4].content, Some("user 5".to_string()));
    }

    #[test]
    fn test_short_history() {
        let history = vec![
            Message::user("system"),
            Message::assistant("assistant 1"),
            Message::user("user 2")
        ];

        let memory = SlidingWindow {
            max_messages: 10,
        };

        let result = memory.window(&history);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].content, Some("system".to_string()));
        assert_eq!(result[1].content, Some("assistant 1".to_string()));
        assert_eq!(result[2].content, Some("user 2".to_string()));
    }

    #[test]
    fn no_system_n_messages() {
        let history = vec![
            Message::user("user 1"),
            Message::assistant("assistant 2"),
            Message::user("user 3"),
            Message::assistant("assistant 4"),
            Message::user("user 5")
        ];

        let memory = SlidingWindow {
            max_messages: 2,
        };

        let result = memory.window(&history);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content, Some("assistant 4".to_string()));
        assert_eq!(result[1].content, Some("user 5".to_string()));
    }

    #[test]
    fn test_includes_assistant_when_starting_with_tool() {
        let history = vec![
            Message::system("system"),
            Message::user("user 1"),
            Message::assistant("assistant 1"),
            Message::user("user 2"),
            Message::assistant("assistant 2"),
            Message::assistant("assistant 3"),
            Message::user("user 3"),
            Message::tool("tool 4", "calculator"),
            Message::assistant("assistant 4"),
            Message::user("user 5")
        ];

        let memory = SlidingWindow {
            max_messages: 3,
        };

        let result = memory.window(&history);

        assert_eq!(result.len(), 5);
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[0].content, Some("system".to_string()));
        assert_eq!(result[1].content, Some("assistant 3".to_string()));
        assert_eq!(result[2].content, Some("tool 4".to_string()));
        assert_eq!(result[3].content, Some("assistant 4".to_string()));
        assert_eq!(result[4].content, Some("user 5".to_string()));
    }
}