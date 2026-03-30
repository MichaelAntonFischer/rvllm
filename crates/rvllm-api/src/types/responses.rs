//! Types for the OpenAI Responses API.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::error::ApiError;
use crate::types::request::ChatMessage;

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_store() -> bool {
    true
}

fn default_parallel_tool_calls() -> bool {
    true
}

/// Raw `input` payload for a Responses API request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Items(Vec<serde_json::Value>),
}

/// Specific tool choice forcing a named function.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseSpecificToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String,
    pub name: String,
}

/// Backwards-compatible nested tool choice form.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseNestedSpecificToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String,
    pub function: ResponseSpecificToolChoiceFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseSpecificToolChoiceFunction {
    pub name: String,
}

/// Tool choice for Responses requests and responses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(untagged)]
pub enum ResponseToolChoice {
    Mode(String),
    Specific(ResponseSpecificToolChoice),
    NestedSpecific(ResponseNestedSpecificToolChoice),
}

impl ResponseToolChoice {
    pub fn validate(&self) -> Result<(), ApiError> {
        match self {
            Self::Mode(mode) => {
                if !["auto", "none", "required"].contains(&mode.as_str()) {
                    return Err(ApiError::InvalidRequest(format!(
                        "invalid tool_choice mode '{}', expected auto/none/required",
                        mode
                    )));
                }
                Ok(())
            }
            Self::Specific(choice) => {
                if choice.choice_type != "function" {
                    return Err(ApiError::InvalidRequest(format!(
                        "unsupported tool_choice type '{}'",
                        choice.choice_type
                    )));
                }
                if choice.name.is_empty() {
                    return Err(ApiError::InvalidRequest(
                        "tool_choice function name must not be empty".into(),
                    ));
                }
                Ok(())
            }
            Self::NestedSpecific(choice) => {
                if choice.choice_type != "function" {
                    return Err(ApiError::InvalidRequest(format!(
                        "unsupported tool_choice type '{}'",
                        choice.choice_type
                    )));
                }
                if choice.function.name.is_empty() {
                    return Err(ApiError::InvalidRequest(
                        "tool_choice function name must not be empty".into(),
                    ));
                }
                Ok(())
            }
        }
    }

    pub fn is_none_mode(&self) -> bool {
        matches!(self, Self::Mode(mode) if mode == "none")
    }

    pub fn forced_tool_name(&self) -> Option<&str> {
        match self {
            Self::Specific(choice) => Some(choice.name.as_str()),
            Self::NestedSpecific(choice) => Some(choice.function.name.as_str()),
            Self::Mode(_) => None,
        }
    }
}

/// Supported custom function tool definition for Responses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseFunctionTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl ResponseFunctionTool {
    pub fn from_value(value: &serde_json::Value) -> Result<Self, ApiError> {
        let tool: Self = serde_json::from_value(value.clone())?;
        if tool.tool_type != "function" {
            return Err(ApiError::InvalidRequest(format!(
                "responses tool type '{}' is not supported yet",
                tool.tool_type
            )));
        }
        if tool.name.is_empty() {
            return Err(ApiError::InvalidRequest(
                "responses function tool name must not be empty".into(),
            ));
        }
        Ok(tool)
    }

    pub fn to_tool_definition(&self) -> rvllm_tokenizer::ToolDefinition {
        rvllm_tokenizer::ToolDefinition {
            tool_type: self.tool_type.clone(),
            function: rvllm_tokenizer::FunctionDefinition {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: self
                    .parameters
                    .as_ref()
                    .and_then(|p| serde_json::from_value(p.clone()).ok()),
            },
        }
    }
}

/// Create a model response request body.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateResponseRequest {
    pub model: String,
    #[serde(default)]
    pub input: Option<ResponseInput>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_store")]
    pub store: bool,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    #[serde(default)]
    pub background: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<ResponseToolChoice>,
    #[serde(default = "default_parallel_tool_calls")]
    pub parallel_tool_calls: bool,
    #[serde(default)]
    pub text: Option<serde_json::Value>,
    #[serde(default)]
    pub reasoning: Option<serde_json::Value>,
    #[serde(default)]
    pub conversation: Option<serde_json::Value>,
    #[serde(default)]
    pub include: Option<Vec<String>>,
    #[serde(default)]
    pub truncation: Option<String>,
}

/// A text input part inside a message item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseInputTextPart {
    pub text: String,
}

impl ResponseInputTextPart {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// An image input part inside a message item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseInputImagePart {
    pub image_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ResponseInputImagePart {
    pub fn new(image_url: impl Into<String>, detail: Option<String>) -> Self {
        Self {
            image_url: image_url.into(),
            detail,
        }
    }
}

/// A normalized content part inside a message item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(tag = "type")]
pub enum ResponseInputContentPart {
    #[serde(rename = "input_text")]
    InputText(ResponseInputTextPart),
    #[serde(rename = "input_image")]
    InputImage(ResponseInputImagePart),
}

impl ResponseInputContentPart {
    pub fn input_text(text: impl Into<String>) -> Self {
        Self::InputText(ResponseInputTextPart::new(text))
    }

    pub fn input_image(image_url: impl Into<String>, detail: Option<String>) -> Self {
        Self::InputImage(ResponseInputImagePart::new(image_url, detail))
    }

    fn to_prompt_text(&self) -> String {
        match self {
            Self::InputText(part) => part.text.clone(),
            Self::InputImage(part) => match part.detail.as_deref() {
                Some(detail) => format!(
                    "[input_image url={} detail={}]",
                    part.image_url, detail
                ),
                None => format!("[input_image url={}]", part.image_url),
            },
        }
    }
}

/// A message item stored in Responses input.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseInputMessage {
    pub role: String,
    pub content: Vec<ResponseInputContentPart>,
}

impl ResponseInputMessage {
    pub fn new(role: impl Into<String>, content: Vec<ResponseInputContentPart>) -> Self {
        Self {
            role: role.into(),
            content,
        }
    }

    pub fn to_chat_message(&self) -> ChatMessage {
        ChatMessage {
            role: self.role.clone(),
            content: self
                .content
                .iter()
                .map(ResponseInputContentPart::to_prompt_text)
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// Tool output item supplied back to the model on a follow-up turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseFunctionCallOutputItem {
    pub call_id: String,
    pub output: serde_json::Value,
}

impl ResponseFunctionCallOutputItem {
    pub fn output_text(&self) -> String {
        match &self.output {
            serde_json::Value::String(text) => text.clone(),
            other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
        }
    }
}

/// Supported normalized input items.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(tag = "type")]
pub enum ResponseInputItem {
    #[serde(rename = "message")]
    Message(ResponseInputMessage),
    #[serde(rename = "function_call_output")]
    FunctionCallOutput(ResponseFunctionCallOutputItem),
}

/// Details about cached prompt tokens.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseInputTokensDetails {
    pub cached_tokens: usize,
}

/// Details about generated output tokens.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseOutputTokensDetails {
    pub reasoning_tokens: usize,
}

/// Token accounting for a Response object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseUsage {
    pub input_tokens: usize,
    pub input_tokens_details: ResponseInputTokensDetails,
    pub output_tokens: usize,
    pub output_tokens_details: ResponseOutputTokensDetails,
    pub total_tokens: usize,
}

impl ResponseUsage {
    pub fn from_request_output(output: &rvllm_core::prelude::RequestOutput) -> Self {
        let output_tokens = output
            .outputs
            .iter()
            .map(|completion| completion.token_ids.len())
            .sum();
        let input_tokens = output.prompt_token_ids.len();
        Self {
            input_tokens,
            input_tokens_details: ResponseInputTokensDetails { cached_tokens: 0 },
            output_tokens,
            output_tokens_details: ResponseOutputTokensDetails {
                reasoning_tokens: 0,
            },
            total_tokens: input_tokens + output_tokens,
        }
    }
}

/// Text format returned from a response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseTextFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Text settings returned from a response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseTextConfig {
    pub format: ResponseTextFormat,
}

impl Default for ResponseTextConfig {
    fn default() -> Self {
        Self {
            format: ResponseTextFormat {
                format_type: "text".to_string(),
                name: None,
                schema: None,
                strict: None,
            },
        }
    }
}

/// Reasoning summary object present on the top-level response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema, Default)]
pub struct ResponseReasoningSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<serde_json::Value>,
}

/// Output text content returned by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseOutputText {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: String,
    pub annotations: Vec<serde_json::Value>,
}

impl ResponseOutputText {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            part_type: "output_text".to_string(),
            text: text.into(),
            annotations: Vec::new(),
        }
    }
}

/// Assistant message item returned in `output`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseOutputMessage {
    pub id: String,
    pub status: String,
    pub role: String,
    pub content: Vec<ResponseOutputText>,
}

impl ResponseOutputMessage {
    pub fn in_progress(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            status: "in_progress".to_string(),
            role: "assistant".to_string(),
            content: Vec::new(),
        }
    }

    pub fn completed(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            status: "completed".to_string(),
            role: "assistant".to_string(),
            content: vec![ResponseOutputText::new(text)],
        }
    }

    pub fn to_chat_message(&self) -> ChatMessage {
        ChatMessage {
            role: self.role.clone(),
            content: self.content.iter().map(|part| part.text.as_str()).collect(),
        }
    }
}

/// Function call item returned in `output`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseFunctionCallItem {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub status: String,
}

impl ResponseFunctionCallItem {
    pub fn in_progress(
        id: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            call_id: call_id.into(),
            name: name.into(),
            arguments: String::new(),
            status: "in_progress".to_string(),
        }
    }

    pub fn completed(
        id: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            call_id: call_id.into(),
            name: name.into(),
            arguments: arguments.into(),
            status: "completed".to_string(),
        }
    }
}

/// Supported normalized output items.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
#[serde(tag = "type")]
pub enum ResponseOutputItem {
    #[serde(rename = "message")]
    Message(ResponseOutputMessage),
    #[serde(rename = "function_call")]
    FunctionCall(ResponseFunctionCallItem),
}

/// Response object returned by the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseObject {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,
    pub model: String,
    pub output: Vec<ResponseOutputItem>,
    pub parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    pub reasoning: ResponseReasoningSummary,
    pub store: bool,
    pub temperature: f32,
    pub text: ResponseTextConfig,
    pub tool_choice: ResponseToolChoice,
    pub tools: Vec<serde_json::Value>,
    pub top_p: f32,
    pub truncation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
    pub metadata: BTreeMap<String, String>,
}

impl ResponseObject {
    #[allow(clippy::too_many_arguments)]
    pub fn in_progress(
        id: impl Into<String>,
        model: impl Into<String>,
        instructions: Option<String>,
        max_output_tokens: Option<usize>,
        previous_response_id: Option<String>,
        store: bool,
        temperature: f32,
        top_p: f32,
        metadata: BTreeMap<String, String>,
        reasoning: ResponseReasoningSummary,
        parallel_tool_calls: bool,
        text: ResponseTextConfig,
        tool_choice: ResponseToolChoice,
        tools: Vec<serde_json::Value>,
    ) -> Self {
        Self {
            id: id.into(),
            object: "response".to_string(),
            created_at: now_seconds(),
            status: "in_progress".to_string(),
            completed_at: None,
            error: None,
            incomplete_details: None,
            instructions,
            max_output_tokens,
            model: model.into(),
            output: Vec::new(),
            parallel_tool_calls,
            previous_response_id,
            reasoning,
            store,
            temperature,
            text,
            tool_choice,
            tools,
            top_p,
            truncation: "disabled".to_string(),
            usage: None,
            metadata,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn completed(
        id: impl Into<String>,
        model: impl Into<String>,
        instructions: Option<String>,
        max_output_tokens: Option<usize>,
        previous_response_id: Option<String>,
        store: bool,
        temperature: f32,
        top_p: f32,
        metadata: BTreeMap<String, String>,
        output: Vec<ResponseOutputItem>,
        usage: ResponseUsage,
        reasoning: ResponseReasoningSummary,
        parallel_tool_calls: bool,
        text: ResponseTextConfig,
        tool_choice: ResponseToolChoice,
        tools: Vec<serde_json::Value>,
    ) -> Self {
        Self {
            id: id.into(),
            object: "response".to_string(),
            created_at: now_seconds(),
            status: "completed".to_string(),
            completed_at: Some(now_seconds()),
            error: None,
            incomplete_details: None,
            instructions,
            max_output_tokens,
            model: model.into(),
            output,
            parallel_tool_calls,
            previous_response_id,
            reasoning,
            store,
            temperature,
            text,
            tool_choice,
            tools,
            top_p,
            truncation: "disabled".to_string(),
            usage: Some(usage),
            metadata,
        }
    }
}

/// Response list wrapper used by `/input_items`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ResponseInputItemsList {
    pub object: String,
    pub data: Vec<ResponseInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
    pub has_more: bool,
}

impl CreateResponseRequest {
    pub fn validate(&self) -> Result<(), ApiError> {
        if self.model.is_empty() {
            return Err(ApiError::InvalidRequest("model is required".into()));
        }
        if self.background == Some(true) {
            if self.stream {
                return Err(ApiError::InvalidRequest(
                    "background responses do not support streaming".into(),
                ));
            }
            if !self.store {
                return Err(ApiError::InvalidRequest(
                    "background responses require store=true".into(),
                ));
            }
        }
        let tools = self.normalize_function_tools()?;
        if let Some(tool_choice) = &self.tool_choice {
            tool_choice.validate()?;
            if tools.is_empty() && !tool_choice.is_none_mode() {
                return Err(ApiError::InvalidRequest(
                    "tool_choice requires at least one function tool".into(),
                ));
            }
            if let Some(forced_name) = tool_choice.forced_tool_name() {
                if !tools.iter().any(|tool| tool.name == forced_name) {
                    return Err(ApiError::InvalidRequest(format!(
                        "tool_choice references unknown tool '{}'",
                        forced_name
                    )));
                }
            }
        }
        let _ = self.normalize_text_config()?;
        let _ = self.normalize_reasoning_config()?;
        if self.conversation.is_some() {
            return Err(ApiError::InvalidRequest(
                "conversation state objects are not supported on /v1/responses yet".into(),
            ));
        }
        let _ = self.normalize_include()?;
        if let Some(truncation) = &self.truncation {
            if truncation != "disabled" {
                return Err(ApiError::InvalidRequest(
                    "only truncation='disabled' is supported".into(),
                ));
            }
        }
        if let Some(max_output_tokens) = self.max_output_tokens {
            if max_output_tokens == 0 {
                return Err(ApiError::InvalidRequest(
                    "max_output_tokens must be greater than 0".into(),
                ));
            }
        }
        if self.temperature < 0.0 || self.temperature > 2.0 {
            return Err(ApiError::InvalidRequest(
                "temperature must be between 0.0 and 2.0".into(),
            ));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(ApiError::InvalidRequest(
                "top_p must be between 0.0 and 1.0".into(),
            ));
        }
        if self.input.is_none() && self.previous_response_id.is_none() {
            return Err(ApiError::InvalidRequest(
                "input is required unless previous_response_id is provided".into(),
            ));
        }
        Ok(())
    }

    pub fn normalize_function_tools(&self) -> Result<Vec<ResponseFunctionTool>, ApiError> {
        self.tools
            .as_ref()
            .map(|tools| tools.iter().map(ResponseFunctionTool::from_value).collect())
            .unwrap_or_else(|| Ok(Vec::new()))
    }

    pub fn effective_tool_choice(&self) -> ResponseToolChoice {
        if let Some(choice) = &self.tool_choice {
            return choice.clone();
        }
        if self.tools_enabled() {
            ResponseToolChoice::Mode("auto".to_string())
        } else {
            ResponseToolChoice::Mode("none".to_string())
        }
    }

    pub fn tools_enabled(&self) -> bool {
        self.tools.as_ref().is_some_and(|tools| !tools.is_empty())
            && !matches!(self.tool_choice.as_ref(), Some(choice) if choice.is_none_mode())
    }

    pub fn to_sampling_params(&self) -> rvllm_core::prelude::SamplingParams {
        let response_format = self
            .to_response_format()
            .unwrap_or(rvllm_core::prelude::ResponseFormat::Text);
        rvllm_core::prelude::SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_output_tokens.unwrap_or(256),
            response_format,
            ..Default::default()
        }
    }

    pub fn normalize_text_config(&self) -> Result<ResponseTextConfig, ApiError> {
        let Some(text) = &self.text else {
            return Ok(ResponseTextConfig::default());
        };
        let Some(map) = text.as_object() else {
            return Err(ApiError::InvalidRequest(
                "responses text must be an object".into(),
            ));
        };
        let Some(format) = map.get("format") else {
            return Ok(ResponseTextConfig::default());
        };
        let Some(format_map) = format.as_object() else {
            return Err(ApiError::InvalidRequest(
                "responses text.format must be an object".into(),
            ));
        };
        let format_type = format_map
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                ApiError::InvalidRequest("responses text.format.type is required".into())
            })?;

        match format_type {
            "text" => Ok(ResponseTextConfig::default()),
            "json_object" => Ok(ResponseTextConfig {
                format: ResponseTextFormat {
                    format_type: "json_object".to_string(),
                    name: None,
                    schema: None,
                    strict: None,
                },
            }),
            "json_schema" => {
                let schema = format_map.get("schema").cloned().ok_or_else(|| {
                    ApiError::InvalidRequest(
                        "responses text.format.schema is required for json_schema".into(),
                    )
                })?;
                let name = format_map
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
                let strict = format_map
                    .get("strict")
                    .and_then(serde_json::Value::as_bool);
                Ok(ResponseTextConfig {
                    format: ResponseTextFormat {
                        format_type: "json_schema".to_string(),
                        name,
                        schema: Some(schema),
                        strict,
                    },
                })
            }
            other => Err(ApiError::InvalidRequest(format!(
                "responses text.format.type '{}' is not supported yet",
                other
            ))),
        }
    }

    pub fn to_response_format(&self) -> Result<rvllm_core::prelude::ResponseFormat, ApiError> {
        let text = self.normalize_text_config()?;
        match text.format.format_type.as_str() {
            "text" => Ok(rvllm_core::prelude::ResponseFormat::Text),
            "json_object" => Ok(rvllm_core::prelude::ResponseFormat::JsonObject),
            "json_schema" => Ok(rvllm_core::prelude::ResponseFormat::JsonSchema {
                json_schema: text.format.schema.ok_or_else(|| {
                    ApiError::InvalidRequest(
                        "responses text.format.schema is required for json_schema".into(),
                    )
                })?,
            }),
            other => Err(ApiError::InvalidRequest(format!(
                "responses text.format.type '{}' is not supported yet",
                other
            ))),
        }
    }

    pub fn normalize_reasoning_config(&self) -> Result<ResponseReasoningSummary, ApiError> {
        let Some(reasoning) = &self.reasoning else {
            return Ok(ResponseReasoningSummary::default());
        };
        let Some(map) = reasoning.as_object() else {
            return Err(ApiError::InvalidRequest(
                "responses reasoning must be an object".into(),
            ));
        };

        let mut normalized = ResponseReasoningSummary::default();
        for (key, value) in map {
            match key.as_str() {
                "effort" => {
                    if value.is_null() {
                        continue;
                    }
                    let effort = value.as_str().ok_or_else(|| {
                        ApiError::InvalidRequest(
                            "responses reasoning.effort must be a string".into(),
                        )
                    })?;
                    match effort {
                        "minimal" | "low" | "medium" | "high" => {
                            normalized.effort = Some(effort.to_string());
                        }
                        other => {
                            return Err(ApiError::InvalidRequest(format!(
                                "responses reasoning.effort '{}' is not supported yet",
                                other
                            )));
                        }
                    }
                }
                other => {
                    return Err(ApiError::InvalidRequest(format!(
                        "responses reasoning.{} is not supported yet",
                        other
                    )));
                }
            }
        }

        Ok(normalized)
    }

    pub fn normalize_include(&self) -> Result<Vec<String>, ApiError> {
        const SUPPORTED_INCLUDES: &[&str] = &[
            "code_interpreter_call.outputs",
            "computer_call_output.output.image_url",
            "file_search_call.results",
            "message.input_image.image_url",
            "message.output_text.logprobs",
            "reasoning.encrypted_content",
            "web_search_call.action.sources",
        ];

        let Some(include) = &self.include else {
            return Ok(Vec::new());
        };

        include
            .iter()
            .map(|value| {
                if SUPPORTED_INCLUDES.contains(&value.as_str()) {
                    Ok(value.clone())
                } else {
                    Err(ApiError::InvalidRequest(format!(
                        "responses include value '{}' is not supported yet",
                        value
                    )))
                }
            })
            .collect()
    }

    pub fn normalize_input_items(&self) -> Result<Vec<ResponseInputItem>, ApiError> {
        match &self.input {
            None => Ok(Vec::new()),
            Some(ResponseInput::Text(text)) => {
                if text.is_empty() {
                    return Err(ApiError::InvalidRequest("input must not be empty".into()));
                }
                Ok(vec![ResponseInputItem::Message(ResponseInputMessage::new(
                    "user",
                    vec![ResponseInputContentPart::input_text(text.clone())],
                ))])
            }
            Some(ResponseInput::Items(items)) => {
                if items.is_empty() {
                    return Err(ApiError::InvalidRequest("input must not be empty".into()));
                }
                items.iter().map(normalize_input_item).collect()
            }
        }
    }
}

fn normalize_input_item(value: &serde_json::Value) -> Result<ResponseInputItem, ApiError> {
    let Some(map) = value.as_object() else {
        return Err(ApiError::InvalidRequest(
            "responses input items must be objects".into(),
        ));
    };

    match map
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("message")
    {
        "message" => Ok(ResponseInputItem::Message(normalize_message_item(value)?)),
        "function_call_output" => Ok(ResponseInputItem::FunctionCallOutput(
            normalize_function_call_output_item(value)?,
        )),
        item_type => Err(ApiError::InvalidRequest(format!(
            "responses input item type '{}' is not supported yet",
            item_type
        ))),
    }
}

fn normalize_message_item(value: &serde_json::Value) -> Result<ResponseInputMessage, ApiError> {
    let Some(map) = value.as_object() else {
        return Err(ApiError::InvalidRequest(
            "responses input items must be objects".into(),
        ));
    };

    let role = map
        .get("role")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ApiError::InvalidRequest("responses input items require a role".into()))?;
    if role.is_empty() {
        return Err(ApiError::InvalidRequest(
            "responses input item role must not be empty".into(),
        ));
    }

    let content = map.get("content").ok_or_else(|| {
        ApiError::InvalidRequest("responses input message items require content".into())
    })?;

    Ok(ResponseInputMessage::new(
        role,
        normalize_input_parts(content)?,
    ))
}

fn normalize_function_call_output_item(
    value: &serde_json::Value,
) -> Result<ResponseFunctionCallOutputItem, ApiError> {
    let Some(map) = value.as_object() else {
        return Err(ApiError::InvalidRequest(
            "responses input items must be objects".into(),
        ));
    };
    let call_id = map
        .get("call_id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            ApiError::InvalidRequest("function_call_output items require call_id".into())
        })?;
    if call_id.is_empty() {
        return Err(ApiError::InvalidRequest(
            "function_call_output call_id must not be empty".into(),
        ));
    }
    let output = map.get("output").cloned().ok_or_else(|| {
        ApiError::InvalidRequest("function_call_output items require output".into())
    })?;
    Ok(ResponseFunctionCallOutputItem {
        call_id: call_id.to_string(),
        output,
    })
}

fn normalize_input_parts(
    value: &serde_json::Value,
) -> Result<Vec<ResponseInputContentPart>, ApiError> {
    match value {
        serde_json::Value::String(text) => {
            if text.is_empty() {
                return Err(ApiError::InvalidRequest(
                    "responses input text must not be empty".into(),
                ));
            }
            Ok(vec![ResponseInputContentPart::input_text(text.clone())])
        }
        serde_json::Value::Array(parts) => {
            if parts.is_empty() {
                return Err(ApiError::InvalidRequest(
                    "responses input content must not be empty".into(),
                ));
            }
            parts
                .iter()
                .map(|part| {
                    let Some(part_map) = part.as_object() else {
                        return Err(ApiError::InvalidRequest(
                            "responses input content parts must be objects".into(),
                        ));
                    };
                    let Some(part_type) = part_map.get("type").and_then(serde_json::Value::as_str)
                    else {
                        return Err(ApiError::InvalidRequest(
                            "responses input content parts require a type".into(),
                        ));
                    };
                    match part_type {
                        "input_text" => {
                            let text = part_map
                                .get("text")
                                .and_then(serde_json::Value::as_str)
                                .ok_or_else(|| {
                                    ApiError::InvalidRequest(
                                        "input_text parts require a text field".into(),
                                    )
                                })?;
                            if text.is_empty() {
                                return Err(ApiError::InvalidRequest(
                                    "responses input text must not be empty".into(),
                                ));
                            }
                            Ok(ResponseInputContentPart::input_text(text.to_string()))
                        }
                        "input_image" => {
                            let image_url = part_map
                                .get("image_url")
                                .and_then(serde_json::Value::as_str)
                                .ok_or_else(|| {
                                    ApiError::InvalidRequest(
                                        "input_image parts require an image_url field".into(),
                                    )
                                })?;
                            if image_url.is_empty() {
                                return Err(ApiError::InvalidRequest(
                                    "responses input image_url must not be empty".into(),
                                ));
                            }
                            let detail = part_map
                                .get("detail")
                                .and_then(serde_json::Value::as_str)
                                .map(str::to_string);
                            Ok(ResponseInputContentPart::input_image(
                                image_url.to_string(),
                                detail,
                            ))
                        }
                        _ => Err(ApiError::InvalidRequest(format!(
                            "responses input content part type '{}' is not supported yet",
                            part_type
                        ))),
                    }
                })
                .collect()
        }
        _ => Err(ApiError::InvalidRequest(
            "responses input content must be a string or array of content parts".into(),
        )),
    }
}

fn now_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_string_input() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        };
        let input = req.normalize_input_items().unwrap();
        assert_eq!(
            input,
            vec![ResponseInputItem::Message(ResponseInputMessage::new(
                "user",
                vec![ResponseInputContentPart::input_text("Hello")]
            ))]
        );
    }

    #[test]
    fn normalize_message_item_input() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Items(vec![serde_json::json!({
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": " there"},
                ]
            })])),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        };
        let input = req.normalize_input_items().unwrap();
        match &input[0] {
            ResponseInputItem::Message(message) => {
                assert_eq!(message.content.len(), 2);
                assert_eq!(message.to_chat_message().content, "Hello there");
            }
            _ => panic!("expected message item"),
        }
    }

    #[test]
    fn normalize_function_call_output_input() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Items(vec![serde_json::json!({
                "type": "function_call_output",
                "call_id": "call_123",
                "output": {"ok": true}
            })])),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        };
        let input = req.normalize_input_items().unwrap();
        assert_eq!(
            input,
            vec![ResponseInputItem::FunctionCallOutput(
                ResponseFunctionCallOutputItem {
                    call_id: "call_123".into(),
                    output: serde_json::json!({"ok": true}),
                }
            )]
        );
    }

    #[test]
    fn accepts_input_image_parts() {
        let parts = normalize_input_parts(&serde_json::json!([
            {
                "type": "input_image",
                "image_url": "https://example.com/image.png",
                "detail": "low"
            }
        ]))
        .unwrap();
        assert_eq!(
            parts,
            vec![ResponseInputContentPart::input_image(
                "https://example.com/image.png",
                Some("low".into())
            )]
        );
    }

    #[test]
    fn image_parts_render_to_prompt_marker() {
        let message = ResponseInputMessage::new(
            "user",
            vec![
                ResponseInputContentPart::input_text("Look at "),
                ResponseInputContentPart::input_image(
                    "https://example.com/image.png",
                    Some("high".into()),
                ),
            ],
        );
        assert_eq!(
            message.to_chat_message().content,
            "Look at [input_image url=https://example.com/image.png detail=high]"
        );
    }

    #[test]
    fn request_accepts_supported_include_values() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: Some(vec![
                "message.input_image.image_url".into(),
                "reasoning.encrypted_content".into(),
            ]),
            truncation: None,
        };
        req.validate().unwrap();
        assert_eq!(
            req.normalize_include().unwrap(),
            vec![
                "message.input_image.image_url".to_string(),
                "reasoning.encrypted_content".to_string(),
            ]
        );
    }

    #[test]
    fn request_rejects_unknown_include_values() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: Some(vec!["message.output_text.foo".into()]),
            truncation: None,
        };
        let err = req.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("responses include value 'message.output_text.foo' is not supported yet"));
    }

    #[test]
    fn request_accepts_function_tools() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: Some(vec![serde_json::json!({
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            })]),
            tool_choice: Some(ResponseToolChoice::Mode("auto".into())),
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        };
        assert!(req.validate().is_ok());
        assert!(req.tools_enabled());
    }

    #[test]
    fn request_rejects_built_in_tools() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: Some(vec![serde_json::json!({"type": "web_search_preview"})]),
            tool_choice: None,
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn request_rejects_unknown_forced_tool() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: Some(vec![serde_json::json!({
                "type": "function",
                "name": "get_weather"
            })]),
            tool_choice: Some(ResponseToolChoice::Specific(ResponseSpecificToolChoice {
                choice_type: "function".into(),
                name: "get_time".into(),
            })),
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn request_accepts_json_object_text_format() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: Some(serde_json::json!({
                "format": {"type": "json_object"}
            })),
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        };
        req.validate().unwrap();
        assert_eq!(
            req.to_response_format().unwrap(),
            rvllm_core::prelude::ResponseFormat::JsonObject
        );
    }

    #[test]
    fn request_accepts_json_schema_text_format() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: Some(serde_json::json!({
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"}
                        },
                        "required": ["name"]
                    },
                    "strict": true
                }
            })),
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        };
        req.validate().unwrap();
        let text = req.normalize_text_config().unwrap();
        assert_eq!(text.format.format_type, "json_schema");
        assert_eq!(text.format.name.as_deref(), Some("answer"));
        assert_eq!(text.format.strict, Some(true));
        assert!(matches!(
            req.to_response_format().unwrap(),
            rvllm_core::prelude::ResponseFormat::JsonSchema { .. }
        ));
    }

    #[test]
    fn request_rejects_unknown_text_format() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: Some(serde_json::json!({
                "format": {"type": "xml"}
            })),
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        };
        let err = req.validate().unwrap_err();
        assert!(err.to_string().contains("responses text.format.type 'xml'"));
    }

    #[test]
    fn request_accepts_background_with_store() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: Some(true),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn request_accepts_reasoning_effort() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: None,
            reasoning: Some(serde_json::json!({
                "effort": "medium"
            })),
            conversation: None,
            include: None,
            truncation: None,
        };
        req.validate().unwrap();
        assert_eq!(
            req.normalize_reasoning_config().unwrap(),
            ResponseReasoningSummary {
                effort: Some("medium".into()),
                summary: None,
            }
        );
    }

    #[test]
    fn request_rejects_unsupported_reasoning_field() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: None,
            reasoning: Some(serde_json::json!({
                "summary": "auto"
            })),
            conversation: None,
            include: None,
            truncation: None,
        };
        let err = req.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("responses reasoning.summary is not supported yet"));
    }

    #[test]
    fn request_rejects_unknown_reasoning_effort() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: None,
            reasoning: Some(serde_json::json!({
                "effort": "max"
            })),
            conversation: None,
            include: None,
            truncation: None,
        };
        let err = req.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("responses reasoning.effort 'max' is not supported yet"));
    }

    #[test]
    fn request_rejects_background_without_store() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: false,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: Some(true),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        };
        let err = req.validate().unwrap_err();
        assert!(err.to_string().contains("background responses require store=true"));
    }

    #[test]
    fn request_rejects_background_streaming() {
        let req = CreateResponseRequest {
            model: "test".into(),
            input: Some(ResponseInput::Text("Hello".into())),
            instructions: None,
            max_output_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            stream: true,
            store: true,
            previous_response_id: None,
            metadata: BTreeMap::new(),
            background: Some(true),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        };
        let err = req.validate().unwrap_err();
        assert!(err.to_string().contains("do not support streaming"));
    }

    #[test]
    fn response_usage_maps_request_output() {
        let output = rvllm_core::prelude::RequestOutput {
            request_id: rvllm_core::prelude::RequestId(1),
            prompt: "Hello".into(),
            prompt_token_ids: vec![1, 2, 3],
            prompt_logprobs: None,
            outputs: vec![rvllm_core::prelude::CompletionOutput {
                index: 0,
                text: "world".into(),
                token_ids: vec![4, 5],
                cumulative_logprob: -1.0,
                logprobs: None,
                finish_reason: Some(rvllm_core::prelude::FinishReason::Stop),
            }],
            finished: true,
        };
        let usage = ResponseUsage::from_request_output(&output);
        assert_eq!(usage.input_tokens, 3);
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(usage.total_tokens, 5);
    }
}
