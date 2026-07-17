use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Unknown tool: {0}")]
    NotFound(String),

    #[error("Invalid arguments for tool: {tool} : {args}")]
    InvalidArguments {
        tool: String,
        args: String,
    },

    #[error("Tool execution failed: {0}.")]
    Execution(String)
}

// Loop 1:
// Obs: reads the question from the user then is location was mentioned? 
//      and time reference? All these info is sent to the model
// Think: It decides that cannot say the wheather without location
// Act: 

// Observe: The user sends the question 
// Think:   1. understand what the user wants
//          2. Plan step - agent decides how to solve the user's request before taking any action
//          3. Decide -  agent decides - what other steps the agent will do in act step
// Act: 