//! Minimal Jev (`TypeSafe` System One) HTTP client over ureq.
//!
//! Providers: [`classifier.dev`](https://classifier.dev) by default (no key,
//! free), with `OpenRouter` and `TypeSafe` as failover when their keys exist.
//! Two request shapes: a `state` the model looks at, and a map of typed
//! questions (noul = yes/no probability, choice = distribution over options).
//! Answers come back under the same keys. Retries 429/529/5xx with backoff.

use std::{collections::BTreeMap, fmt, thread, time::Duration};

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Endpoint {
	/// classifier.dev — free zero-shot classification, no key. Default.
	Classifier,
	Openrouter,
	Typesafe,
}

impl Endpoint {
	pub const fn key_name(self) -> &'static str {
		match self {
			Self::Classifier => "",
			Self::Openrouter => "OPENROUTER_API_KEY",
			Self::Typesafe => "TYPESAFE_API_KEY",
		}
	}

	const fn url(self) -> &'static str {
		match self {
			Self::Classifier => "https://classifier.dev/v1/classify",
			Self::Openrouter => "https://openrouter.ai/api/alpha/decisions",
			Self::Typesafe => "https://api.typesafe.ai/v1/systemone",
		}
	}
}

/// classifier.dev request caps: 20 dimensions per request, 2–100 labels per
/// dimension, 4,000 characters of instructions per dimension with 16,000
/// combined (kept under with margin), and a 32,000-character input.
const MAX_DIMENSIONS: usize = 20;
const MAX_LABELS: usize = 100;
const MAX_INSTRUCTIONS: usize = 4_000;
const MAX_COMBINED_INSTRUCTIONS: usize = 15_000;
const MAX_INPUT_CHARS: usize = 32_000;

struct Provider {
	url:  String,
	auth: String,
	kind: Endpoint,
}

/// Load the provider chain: classifier.dev first (no key), then the keyed
/// providers whose keys exist, in the given order. A pinned provider is
/// required and comes first.
fn providers(
	preferred: Option<Endpoint>,
	mut lookup: impl FnMut(&str) -> Result<String, String>,
) -> Result<Vec<Provider>, String> {
	let mut providers = Vec::new();
	let mut push_keyed =
		|endpoint: Endpoint, required: bool, providers: &mut Vec<Provider>| -> Result<(), String> {
			match lookup(endpoint.key_name()) {
				Ok(key) => {
					providers.push(Provider {
						url:  endpoint.url().into(),
						auth: format!("Bearer {key}"),
						kind: endpoint,
					});
					Ok(())
				},
				Err(e) if required => Err(e),
				Err(_) => Ok(()),
			}
		};
	match preferred {
		Some(Endpoint::Classifier) | None => providers.push(Provider {
			url:  Endpoint::Classifier.url().into(),
			auth: String::new(),
			kind: Endpoint::Classifier,
		}),
		Some(endpoint) => push_keyed(endpoint, true, &mut providers)?,
	}
	match preferred {
		None => {
			push_keyed(Endpoint::Openrouter, false, &mut providers)?;
			push_keyed(Endpoint::Typesafe, false, &mut providers)?;
		},
		Some(Endpoint::Openrouter) => push_keyed(Endpoint::Typesafe, false, &mut providers)?,
		Some(Endpoint::Typesafe) => push_keyed(Endpoint::Openrouter, false, &mut providers)?,
		Some(Endpoint::Classifier) => {},
	}
	Ok(providers)
}
/// Published price: $42 per billion input tokens; output tokens are free.
pub const USD_PER_INPUT_TOKEN: f64 = 42.0 / 1e9;
/// Total retries and failover attempts performed by every client in this
/// process.
pub static RETRIES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Physical HTTP attempts, including question chunks, retries, and failovers.
pub static HTTP_ATTEMPTS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[derive(Serialize, Clone, Debug)]
pub struct NoulCriteria {
	#[serde(rename = "true")]
	pub yes: String,
	#[serde(rename = "false")]
	pub no:  String,
}

#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
	Noul {
		instructions: Value,
		#[serde(skip_serializing_if = "Option::is_none")]
		criteria:     Option<NoulCriteria>,
	},
	Choice {
		instructions: Value,
		/// option name -> description (or null when the name is self-explanatory)
		criteria:     BTreeMap<String, Value>,
	},
}

#[derive(Serialize)]
struct RequestBody<'a> {
	state:     &'a Value,
	model:     &'a str,
	questions: &'a BTreeMap<String, Question>,
}

#[derive(Deserialize, Debug, Default, Clone, Copy)]
pub struct Usage {
	pub input_tokens:  u64,
	pub output_tokens: u64,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
	Noul { noul: f64 },
	Choice { choice: String, probabilities: BTreeMap<String, f64>, confidence: f64 },
	Score { score: f64 },
}

#[derive(Deserialize, Debug, Default)]
pub struct Response {
	pub model:   String,
	pub answers: BTreeMap<String, Answer>,
	pub usage:   Usage,
}

impl Response {
	pub fn noul(&self, key: &str) -> Option<f64> {
		match self.answers.get(key) {
			Some(Answer::Noul { noul }) => Some(*noul),
			_ => None,
		}
	}

	pub fn choice(&self, key: &str) -> Option<(&BTreeMap<String, f64>, f64)> {
		match self.answers.get(key) {
			Some(Answer::Choice { probabilities, confidence, .. }) => {
				Some((probabilities, *confidence))
			},
			_ => None,
		}
	}
}

#[derive(Debug, Clone)]
pub enum Error {
	Status(u16, String),
	Transport(String),
	Decode(String),
}

impl Error {
	fn can_failover(&self) -> bool {
		match self {
			Self::Status(status, _) => {
				matches!(status, 401 | 402 | 403 | 408 | 429) || (500..600).contains(status)
			},
			Self::Transport(_) | Self::Decode(_) => true,
		}
	}
}

impl fmt::Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Status(code, body) => write!(f, "HTTP {code}: {body}"),
			Self::Transport(e) => write!(f, "transport: {e}"),
			Self::Decode(e) => write!(f, "decode: {e}"),
		}
	}
}

pub struct Client {
	agent:          ureq::Agent,
	providers:      Vec<Provider>,
	active:         std::sync::atomic::AtomicUsize,
	model:          String,
	max_retries:    u32,
	/// Opt-in limit for independent Noul questions per HTTP request. Each
	/// logical call has at most two chunks in flight, so strategy --parallel P
	/// permits up to 2*P physical requests. Choice and mixed batches remain
	/// intact.
	question_chunk: Option<usize>,
}

impl Client {
	pub fn new(endpoint: Option<Endpoint>, model: String) -> Result<Self, String> {
		let providers = providers(endpoint, crate::env::api_key)?;
		let question_chunk = match std::env::var("JEGREP_QUESTION_CHUNK") {
			Ok(value) => Some(
				value
					.parse::<usize>()
					.ok()
					.filter(|n| *n > 0)
					.ok_or("JEGREP_QUESTION_CHUNK must be a positive integer")?,
			),
			Err(std::env::VarError::NotPresent) => None,
			Err(_) => return Err("JEGREP_QUESTION_CHUNK must be a positive integer".into()),
		};
		let agent = ureq::Agent::config_builder()
			.http_status_as_error(false)
			.timeout_global(Some(Duration::from_secs(120)))
			.build()
			.new_agent();
		Ok(Self {
			agent,
			providers,
			active: std::sync::atomic::AtomicUsize::new(0),
			model,
			max_retries: 6,
			question_chunk,
		})
	}

	/// Judge one state against a batch of questions. Provider dispatch and
	/// question chunking happen per attempt, so a failover may switch both.
	pub fn system_one(
		&self,
		state: &Value,
		questions: &BTreeMap<String, Question>,
	) -> Result<Response, Error> {
		use std::sync::atomic::Ordering;
		let first = self.active.load(Ordering::Relaxed);
		let mut order: Vec<usize> = (0..self.providers.len())
			.map(|i| (first + i) % self.providers.len())
			.collect();
		// classifier.dev caps a dimension at 100 labels; a bigger Choice needs
		// a keyed provider that accepts up to 255 options.
		if questions
			.values()
			.any(|q| matches!(q, Question::Choice { criteria, .. } if criteria.len() > MAX_LABELS))
		{
			if order
				.iter()
				.all(|&i| self.providers[i].kind == Endpoint::Classifier)
			{
				return Err(Error::Status(
					400,
					"a choice with more than 100 options requires OPENROUTER_API_KEY or \
					 TYPESAFE_API_KEY (classifier.dev caps a dimension at 100 labels)"
						.into(),
				));
			}
			order.sort_by_key(|&i| self.providers[i].kind == Endpoint::Classifier);
		}
		let mut result = self.send(order[0], state, questions, order.len() == 1);
		for &fallback in &order[1..] {
			match &result {
				Err(error) if error.can_failover() => {
					RETRIES.fetch_add(1, Ordering::Relaxed);
					result = self.send(fallback, state, questions, true);
					if result.is_ok() {
						self.active.store(fallback, Ordering::Relaxed);
					}
				},
				_ => break,
			}
		}
		result
	}

	fn send(
		&self,
		provider: usize,
		state: &Value,
		questions: &BTreeMap<String, Question>,
		retry: bool,
	) -> Result<Response, Error> {
		let provider = &self.providers[provider];
		match provider.kind {
			Endpoint::Classifier => self.classify(provider, state, questions, retry),
			_ => self.system_one_keyed(provider, state, questions, retry),
		}
	}

	/// The keyed System One request shape, optionally split into independent
	/// Noul chunks by `JEGREP_QUESTION_CHUNK`.
	fn system_one_keyed(
		&self,
		provider: &Provider,
		state: &Value,
		questions: &BTreeMap<String, Question>,
		retry: bool,
	) -> Result<Response, Error> {
		let chunks: Vec<BTreeMap<String, Question>> =
			match self.question_chunk.filter(|n| questions.len() > *n) {
				// Choice probabilities depend on the complete option set. Leave
				// mixed batches together too, rather than assuming independence
				// across types.
				Some(chunk_size)
					if questions
						.values()
						.all(|q| matches!(q, Question::Noul { .. })) =>
				{
					questions
						.iter()
						.map(|(key, question)| ((*key).clone(), (*question).clone()))
						.collect::<Vec<_>>()
						.chunks(chunk_size)
						.map(|chunk| chunk.iter().cloned().collect())
						.collect()
				},
				_ => vec![questions.clone()],
			};
		merge_chunks(&chunks, |chunk| {
			let body = RequestBody { state, model: &self.model, questions: chunk };
			if std::env::var_os("JEGREP_DUMP").is_some()
				&& let Ok(text) = serde_json::to_string_pretty(&body)
			{
				let shown: String = text.chars().take(6000).collect();
				crate::ui::diagnostic(&format!(
					"──── request ({} bytes) ────\n{shown}{}\n────",
					text.len(),
					if text.len() > 6000 { "\n…" } else { "" }
				));
			}
			let (status, text) = self.post(provider, &serde_json::to_value(&body).unwrap(), retry)?;
			if status != 200 {
				return Err(Error::Status(status, truncate(&text, 300)));
			}
			serde_json::from_str(&text).map_err(|e| Error::Decode(e.to_string()))
		})
	}

	/// classifier.dev: the shared state becomes the single input text and every
	/// question becomes an independent dimension. Free and keyless, so usage
	/// stays at zero tokens.
	fn classify(
		&self,
		provider: &Provider,
		state: &Value,
		questions: &BTreeMap<String, Question>,
		retry: bool,
	) -> Result<Response, Error> {
		let input = state_text(state);
		let chunks = dimension_chunks(questions);
		merge_chunks(&chunks, |chunk| {
			let dimensions: BTreeMap<&str, Value> = chunk
				.iter()
				.map(|(key, question)| {
					(
						key.as_str(),
						serde_json::json!({
							"labels": labels_of(question),
							"instructions": question_text(question),
						}),
					)
				})
				.collect();
			let body =
				serde_json::json!({ "items": [input], "tier": "fast", "dimensions": dimensions });
			if let Ok(text) = serde_json::to_string_pretty(&body)
				&& std::env::var_os("JEGREP_DUMP").is_some()
			{
				let shown: String = text.chars().take(6000).collect();
				crate::ui::diagnostic(&format!(
					"──── request ({} bytes) ────\n{shown}{}\n────",
					text.len(),
					if text.len() > 6000 { "\n…" } else { "" }
				));
			}
			let (status, text) = self.post(provider, &body, retry)?;
			if status != 200 {
				return Err(Error::Status(status, truncate(&text, 300)));
			}
			let parsed: ClassifierResponse =
				serde_json::from_str(&text).map_err(|e| Error::Decode(e.to_string()))?;
			parsed.into_response(chunk)
		})
	}

	/// One physical HTTP POST with retry/backoff. Returns the status and body
	/// for the caller to interpret; transport failures that are not clearly
	/// permanent are retried here.
	fn post(&self, provider: &Provider, body: &Value, retry: bool) -> Result<(u16, String), Error> {
		use std::sync::atomic::Ordering;
		let mut attempt = 0u32;
		loop {
			HTTP_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
			match self
				.agent
				.post(&provider.url)
				.header("Authorization", &provider.auth)
				.header("Content-Type", "application/json")
				.send_json(body)
			{
				Ok(mut resp) => {
					let status = resp.status().as_u16();
					let retry_after = resp
						.headers()
						.get("retry-after")
						.and_then(|v| v.to_str().ok())
						.and_then(|s| s.trim().parse::<f64>().ok());
					let text = resp.body_mut().read_to_string().unwrap_or_default();
					let transient = status == 429 || status == 529 || (500..600).contains(&status);
					if retry && transient && attempt < self.max_retries {
						attempt += 1;
						RETRIES.fetch_add(1, Ordering::Relaxed);
						if !WARNED.swap(true, Ordering::Relaxed) {
							crate::ui::diagnostic(&format!(
								"  ! jev: HTTP {status}, backing off (further retries are counted \
								 silently; see footer)"
							));
						}
						thread::sleep(backoff(attempt, retry_after));
						continue;
					}
					return Ok((status, text));
				},
				Err(e) => {
					let transient =
						!matches!(e, ureq::Error::BadUri(_) | ureq::Error::Http(_) | ureq::Error::Tls(_));
					if retry && transient && attempt < self.max_retries {
						attempt += 1;
						RETRIES.fetch_add(1, Ordering::Relaxed);
						thread::sleep(backoff(attempt, None));
						continue;
					}
					return Err(Error::Transport(e.to_string()));
				},
			}
		}
	}
}

// ── classifier.dev translation ──────────────────────────────────────────────

/// Split questions into independent requests and merge the answers. Chunks
/// run in waves of two so fan-out stays bounded; a failed chunk aborts the
/// remaining waves.
fn merge_chunks(
	chunks: &[BTreeMap<String, Question>],
	request: impl Fn(&BTreeMap<String, Question>) -> Result<Response, Error> + Sync,
) -> Result<Response, Error> {
	let mut merged = Response::default();
	for wave in chunks.chunks(2) {
		let responses = thread::scope(|scope| {
			wave
				.iter()
				.map(|chunk| scope.spawn(|| request(chunk)))
				.collect::<Vec<_>>()
				.into_iter()
				.map(|worker| worker.join().expect("question chunk worker panicked"))
				.collect::<Vec<_>>()
		});
		for response in responses {
			let response = response?;
			if merged.model.is_empty() {
				merged.model = response.model;
			}
			merged.usage.input_tokens += response.usage.input_tokens;
			merged.usage.output_tokens += response.usage.output_tokens;
			for (key, answer) in response.answers {
				if merged.answers.insert(key.clone(), answer).is_some() {
					return Err(Error::Decode(format!(
						"duplicate answer key across question chunks: {key}"
					)));
				}
			}
		}
	}
	Ok(merged)
}

/// The serialized state is the model-facing input, capped at the API limit.
fn state_text(state: &Value) -> String {
	let text = serde_json::to_string(state).unwrap_or_default();
	if text.chars().count() > MAX_INPUT_CHARS {
		text.chars().take(MAX_INPUT_CHARS).collect()
	} else {
		text
	}
}

/// Noul questions classify over yes/no; the yes share is the probability.
fn labels_of(question: &Question) -> Vec<&str> {
	match question {
		Question::Noul { .. } => vec!["yes", "no"],
		Question::Choice { criteria, .. } => criteria.keys().map(String::as_str).collect(),
	}
}

/// Flatten a question into classifier instructions, folding the criteria in —
/// the API only carries one instructions string per dimension.
fn question_text(question: &Question) -> String {
	let (instructions, detail) = match question {
		Question::Noul { instructions, criteria } => (
			instructions,
			criteria
				.as_ref()
				.map(|c| format!("\n\n\"yes\" means: {}\n\"no\" means: {}", c.yes, c.no)),
		),
		Question::Choice { instructions, criteria } => {
			let options: String = criteria
				.iter()
				.map(|(name, desc)| match desc {
					Value::String(d) => format!("\n- {name}: {d}"),
					Value::Null => format!("\n- {name}"),
					other => format!("\n- {name}: {other}"),
				})
				.collect();
			(instructions, Some(format!("\n\nOptions:{options}")))
		},
	};
	let mut text = match instructions {
		Value::String(s) => s.clone(),
		other => other.to_string(),
	};
	if let Some(detail) = detail {
		text.push_str(&detail);
	}
	if text.chars().count() > MAX_INSTRUCTIONS {
		text.chars().take(MAX_INSTRUCTIONS).collect()
	} else {
		text
	}
}

/// Chunk questions into classifier requests: at most 20 dimensions, and at
/// most ~15,000 combined instruction characters (API cap 16,000).
fn dimension_chunks(questions: &BTreeMap<String, Question>) -> Vec<BTreeMap<String, Question>> {
	let mut chunks = Vec::new();
	let mut current = BTreeMap::new();
	let mut combined = 0usize;
	for (key, question) in questions {
		let size = question_text(question).len();
		if !current.is_empty()
			&& (current.len() >= MAX_DIMENSIONS || combined + size > MAX_COMBINED_INSTRUCTIONS)
		{
			chunks.push(std::mem::take(&mut current));
			combined = 0;
		}
		combined += size;
		current.insert(key.clone(), question.clone());
	}
	if !current.is_empty() {
		chunks.push(current);
	}
	chunks
}

#[derive(Deserialize, Debug, Default)]
struct ClassifierResponse {
	model:   String,
	results: Vec<ClassifierResult>,
}

#[derive(Deserialize, Debug)]
struct ClassifierResult {
	dimensions: BTreeMap<String, ClassifierAnswer>,
}

#[derive(Deserialize, Debug)]
struct ClassifierAnswer {
	#[serde(default)]
	label:      Option<String>,
	#[serde(default)]
	confidence: Option<f64>,
	#[serde(default)]
	scores:     Option<BTreeMap<String, Option<f64>>>,
}

impl ClassifierResponse {
	fn into_response(self, questions: &BTreeMap<String, Question>) -> Result<Response, Error> {
		let dimensions = &self
			.results
			.into_iter()
			.next()
			.ok_or_else(|| Error::Decode("classifier response has no results".into()))?
			.dimensions;
		let mut answers = BTreeMap::new();
		for (key, question) in questions {
			let answer = dimensions
				.get(key)
				.ok_or_else(|| Error::Decode(format!("classifier response is missing {key}")))?;
			answers.insert(key.clone(), match question {
				Question::Noul { .. } => Answer::Noul {
					noul: answer
						.scores
						.as_ref()
						.and_then(|s| s.get("yes").copied().flatten())
						.or(match answer.label.as_deref() {
							Some("yes") => Some(1.0),
							Some("no") => Some(0.0),
							_ => None,
						})
						.unwrap_or(0.5),
				},
				Question::Choice { .. } => Answer::Choice {
					choice:        answer.label.clone().unwrap_or_default(),
					probabilities: answer
						.scores
						.as_ref()
						.map(|s| {
							s.iter()
								.filter_map(|(k, v)| v.map(|v| (k.clone(), v)))
								.collect()
						})
						.unwrap_or_default(),
					confidence:    answer.confidence.unwrap_or(0.0),
				},
			});
		}
		Ok(Response { model: self.model, answers, usage: Usage::default() })
	}
}

fn backoff(attempt: u32, retry_after: Option<f64>) -> Duration {
	let base = 0.4 * 2f64.powi(attempt as i32 - 1);
	// cheap deterministic jitter from the address of a stack value
	let jitter = (&attempt as *const u32 as usize % 250) as f64 / 1000.0;
	let secs = retry_after.unwrap_or(0.0).max(base) + jitter;
	Duration::from_secs_f64(secs.min(20.0))
}

fn truncate(s: &str, n: usize) -> String {
	let s = s.trim();
	if s.chars().count() <= n {
		s.to_string()
	} else {
		format!("{}…", s.chars().take(n).collect::<String>())
	}
}

#[cfg(test)]
mod tests {
	use std::{
		io::{Read, Write},
		net::TcpListener,
		sync::atomic::Ordering,
	};

	use super::*;

	#[test]
	fn provider_selection() {
		let both = |name: &str| Ok(name.to_owned());
		let auto = providers(None, both).unwrap();
		assert_eq!(auto[0].kind, Endpoint::Classifier);
		assert_eq!(auto[1].url, Endpoint::Openrouter.url());
		assert_eq!(auto[1].auth, "Bearer OPENROUTER_API_KEY");
		assert_eq!(auto[2].auth, "Bearer TYPESAFE_API_KEY");
		// Pinned keyed providers keep their pair as failover; classifier.dev is
		// dropped from the chain because the pin was explicit.
		let explicit = providers(Some(Endpoint::Typesafe), both).unwrap();
		assert_eq!(explicit[0].kind, Endpoint::Typesafe);
		assert_eq!(explicit[1].kind, Endpoint::Openrouter);
		let pinned = providers(Some(Endpoint::Classifier), both).unwrap();
		assert_eq!(pinned.len(), 1);
		assert_eq!(pinned[0].kind, Endpoint::Classifier);
		// classifier.dev needs no key: it is always available by default.
		let only_typesafe = |name: &str| {
			if name == "TYPESAFE_API_KEY" {
				Ok("direct-key".into())
			} else {
				Err("missing".into())
			}
		};
		let chain = providers(None, only_typesafe).unwrap();
		assert_eq!(chain[0].kind, Endpoint::Classifier);
		assert_eq!(chain[1].url, Endpoint::Typesafe.url());
		assert!(providers(Some(Endpoint::Openrouter), only_typesafe).is_err());
		assert_eq!(providers(None, |_| Err("missing".into())).unwrap()[0].kind, Endpoint::Classifier);
	}

	fn server(status: u16, count: usize, key: &'static str) -> (String, thread::JoinHandle<()>) {
		let listener = TcpListener::bind("127.0.0.1:0").unwrap();
		let url = format!("http://{}", listener.local_addr().unwrap());
		let handle = thread::spawn(move || {
			for _ in 0..count {
				let (mut stream, _) = listener.accept().unwrap();
				stream
					.set_read_timeout(Some(Duration::from_secs(5)))
					.unwrap();
				let mut request = Vec::new();
				let mut buf = [0; 4096];
				loop {
					let n = stream.read(&mut buf).unwrap();
					assert!(n > 0);
					request.extend_from_slice(&buf[..n]);
					if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
						let headers = String::from_utf8_lossy(&request[..end]).to_lowercase();
						let length: usize = headers
							.lines()
							.find_map(|line| line.strip_prefix("content-length:"))
							.unwrap()
							.trim()
							.parse()
							.unwrap();
						if request.len() >= end + 4 + length {
							break;
						}
					}
				}
				let request = String::from_utf8(request).unwrap();
				assert!(
					request
						.to_lowercase()
						.contains(&format!("authorization: bearer {key}"))
				);
				assert!(request.contains("jev-latest"));
				let body = r#"{"model":"jev-latest","answers":{},"usage":{"input_tokens":1,"output_tokens":0}}"#;
				write!(
					stream,
					"HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nContent-Type: \
					 application/json\r\nConnection: close\r\n\r\n{body}",
					body.len()
				)
				.unwrap();
			}
		});
		(url, handle)
	}

	#[test]
	fn failover_uses_matching_key_and_sticks_to_successful_provider() {
		for status in [401, 402, 403, 408, 429, 500, 529] {
			let (primary, p) = server(status, 1, "primary");
			let (fallback, f) = server(200, 2, "fallback");
			let mut client = Client {
				agent:          ureq::Agent::config_builder()
					.http_status_as_error(false)
					.build()
					.new_agent(),
				providers:      vec![],
				active:         std::sync::atomic::AtomicUsize::new(0),
				model:          "jev-latest".into(),
				max_retries:    0,
				question_chunk: None,
			};
			client.providers = vec![
				Provider { url: primary, auth: "Bearer primary".into(), kind: Endpoint::Openrouter },
				Provider { url: fallback, auth: "Bearer fallback".into(), kind: Endpoint::Typesafe },
			];
			client.max_retries = 0;
			for _ in 0..2 {
				client.system_one(&Value::Null, &BTreeMap::new()).unwrap();
			}
			assert_eq!(client.active.load(Ordering::Relaxed), 1);
			p.join().unwrap();
			f.join().unwrap();
		}
	}

	#[test]
	fn request_errors_do_not_fail_over() {
		assert!(!Error::Status(400, String::new()).can_failover());
		assert!(!Error::Status(422, String::new()).can_failover());
		assert!(Error::Transport("connection closed".into()).can_failover());
		assert!(Error::Decode("invalid JSON".into()).can_failover());
	}

	fn noul_questions(count: usize) -> BTreeMap<String, Question> {
		(0..count)
			.map(|i| {
				(format!("q{i}"), Question::Noul {
					instructions: Value::String(format!("Question {i}")),
					criteria:     None,
				})
			})
			.collect()
	}

	fn chunk_client(url: String, question_chunk: Option<usize>) -> Client {
		Client {
			agent: ureq::Agent::config_builder()
				.http_status_as_error(false)
				.timeout_global(Some(Duration::from_secs(5)))
				.build()
				.new_agent(),
			providers: vec![Provider {
				url,
				auth: "Bearer primary".into(),
				kind: Endpoint::Openrouter,
			}],
			active: std::sync::atomic::AtomicUsize::new(0),
			model: "jev-latest".into(),
			max_retries: 0,
			question_chunk,
		}
	}

	// The mock accepts concurrently, records the actual HTTP JSON, and reports
	// peak in-flight requests. A bounded accept deadline makes regressions fail
	// locally instead of hanging forever waiting for an expected request.
	fn question_server(
		count: usize,
		status_for: fn(&Value) -> u16,
	) -> (String, thread::JoinHandle<Vec<Value>>, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
		use std::sync::{Arc, atomic::AtomicUsize};
		let listener = TcpListener::bind("127.0.0.1:0").unwrap();
		listener.set_nonblocking(true).unwrap();
		let url = format!("http://{}", listener.local_addr().unwrap());
		let peak = Arc::new(AtomicUsize::new(0));
		let server_peak = peak.clone();
		let handle = thread::spawn(move || {
			let active = Arc::new(AtomicUsize::new(0));
			let mut workers = Vec::new();
			let deadline = std::time::Instant::now() + Duration::from_secs(5);
			while workers.len() < count {
				let mut stream = match listener.accept() {
					Ok((stream, _)) => stream,
					Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
						assert!(std::time::Instant::now() < deadline, "missing mock HTTP request");
						thread::sleep(Duration::from_millis(1));
						continue;
					},
					Err(e) => panic!("mock accept: {e}"),
				};
				let active = active.clone();
				let peak = server_peak.clone();
				workers.push(thread::spawn(move || {
					stream.set_nonblocking(false).unwrap();
					stream
						.set_read_timeout(Some(Duration::from_secs(5)))
						.unwrap();
					let mut bytes = Vec::new();
					let mut buffer = [0; 4096];
					let request: Value = loop {
						let n = stream.read(&mut buffer).unwrap();
						assert!(n > 0);
						bytes.extend_from_slice(&buffer[..n]);
						if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
							let headers = String::from_utf8_lossy(&bytes[..end]).to_lowercase();
							let length: usize = headers
								.lines()
								.find_map(|line| line.strip_prefix("content-length:"))
								.unwrap()
								.trim()
								.parse()
								.unwrap();
							if bytes.len() >= end + 4 + length {
								break serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
							}
						}
					};
					peak.fetch_max(active.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
					thread::sleep(Duration::from_millis(40));
					let status = status_for(&request);
					let answers: BTreeMap<_, _> = request["questions"]
						.as_object()
						.unwrap()
						.keys()
						.map(|key| (key, serde_json::json!({"type": "noul", "noul": 0.75})))
						.collect();
					let n = answers.len();
					let body = serde_json::json!({"model": "jev-latest", "answers": answers,
                        "usage": {"input_tokens": 100 + n * 10, "output_tokens": n}})
					.to_string();
					write!(
						stream,
						"HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nContent-Type: \
						 application/json\r\nConnection: close\r\n\r\n{body}",
						body.len()
					)
					.unwrap();
					active.fetch_sub(1, Ordering::SeqCst);
					request
				}));
			}
			workers
				.into_iter()
				.map(|worker| worker.join().unwrap())
				.collect()
		});
		(url, handle, peak)
	}

	#[test]
	fn question_chunks_preserve_state_merge_usage_and_bound_concurrency() {
		let (url, server, peak) = question_server(4, |_| 200);
		let client = chunk_client(url, Some(2));
		let state = serde_json::json!({"content": "the same state in every request"});
		let questions = noul_questions(7);
		let response = client.system_one(&state, &questions).unwrap();
		let requests = server.join().unwrap();
		assert_eq!(requests.len(), 4);
		assert_eq!(peak.load(Ordering::SeqCst), 2);
		let mut seen = BTreeMap::new();
		for request in requests {
			assert_eq!(request["state"], state);
			assert_eq!(request["model"], "jev-latest");
			let subset = request["questions"].as_object().unwrap();
			assert!((1..=2).contains(&subset.len()));
			for (key, value) in subset {
				assert!(seen.insert(key.clone(), value.clone()).is_none());
			}
		}
		assert_eq!(serde_json::to_value(&seen).unwrap(), serde_json::to_value(&questions).unwrap());
		assert_eq!(response.model, "jev-latest");
		assert_eq!(response.answers.len(), 7);
		for key in questions.keys() {
			assert_eq!(response.noul(key), Some(0.75));
		}
		assert_eq!(response.usage.input_tokens, 470);
		assert_eq!(response.usage.output_tokens, 7);
	}

	#[test]
	fn default_small_choice_and_mixed_batches_are_not_split() {
		let choice = Question::Choice {
			instructions: Value::String("Choose a file".into()),
			criteria:     (0..6)
				.map(|i| (format!("option{i}"), Value::Null))
				.collect(),
		};
		let mut mixed = noul_questions(4);
		mixed.insert("choose".into(), choice.clone());
		let only_choice = [("choose".into(), choice)].into_iter().collect();
		for (questions, chunk) in [
			(noul_questions(5), None),
			(noul_questions(2), Some(2)),
			(only_choice, Some(1)),
			(mixed, Some(1)),
		] {
			let (url, server, _) = question_server(1, |_| 200);
			chunk_client(url, chunk)
				.system_one(&Value::Null, &questions)
				.unwrap();
			let requests = server.join().unwrap();
			assert_eq!(requests[0]["questions"], serde_json::to_value(&questions).unwrap());
		}
	}

	#[test]
	fn question_chunk_error_is_propagated_without_sending_later_waves() {
		let (url, server, _) = question_server(2, |request| {
			if request["questions"].get("q0").is_some() {
				400
			} else {
				200
			}
		});
		let result = chunk_client(url, Some(1)).system_one(&Value::Null, &noul_questions(5));
		assert!(matches!(result, Err(Error::Status(400, _))));
		assert_eq!(server.join().unwrap().len(), 2);
	}

	#[test]
	fn question_chunks_keep_provider_failover() {
		let (primary, p, _) = question_server(2, |_| 401);
		let (fallback, f, _) = question_server(3, |_| 200);
		let mut client = chunk_client(primary, Some(1));
		client.providers.push(Provider {
			url:  fallback,
			auth: "Bearer fallback".into(),
			kind: Endpoint::Typesafe,
		});
		let response = client.system_one(&Value::Null, &noul_questions(3)).unwrap();
		assert_eq!(response.answers.len(), 3);
		assert_eq!(response.usage.input_tokens, 330);
		assert_eq!(response.usage.output_tokens, 3);
		assert_eq!(client.active.load(Ordering::Relaxed), 1);
		assert_eq!(p.join().unwrap().len(), 2);
		assert_eq!(f.join().unwrap().len(), 3);
	}

	fn classifier_client(url: String) -> Client {
		Client {
			agent:          ureq::Agent::config_builder()
				.http_status_as_error(false)
				.timeout_global(Some(Duration::from_secs(5)))
				.build()
				.new_agent(),
			providers:      vec![Provider { url, auth: String::new(), kind: Endpoint::Classifier }],
			active:         std::sync::atomic::AtomicUsize::new(0),
			model:          "jev-latest".into(),
			max_retries:    0,
			question_chunk: None,
		}
	}

	/// Accepts `count` classifier requests, records the parsed JSON bodies, and
	/// answers every dimension with fixed yes/no scores.
	fn classifier_server(
		count: usize,
		status_for: fn(&Value) -> u16,
	) -> (String, thread::JoinHandle<Vec<Value>>) {
		let listener = TcpListener::bind("127.0.0.1:0").unwrap();
		listener.set_nonblocking(true).unwrap();
		let url = format!("http://{}", listener.local_addr().unwrap());
		let handle = thread::spawn(move || {
			let mut workers = Vec::new();
			let deadline = std::time::Instant::now() + Duration::from_secs(5);
			while workers.len() < count {
				let mut stream = match listener.accept() {
					Ok((stream, _)) => stream,
					Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
						assert!(std::time::Instant::now() < deadline, "missing mock HTTP request");
						thread::sleep(Duration::from_millis(1));
						continue;
					},
					Err(e) => panic!("mock accept: {e}"),
				};
				workers.push(thread::spawn(move || {
					stream.set_nonblocking(false).unwrap();
					stream
						.set_read_timeout(Some(Duration::from_secs(5)))
						.unwrap();
					let mut bytes = Vec::new();
					let mut buffer = [0; 4096];
					let request: Value = loop {
						let n = stream.read(&mut buffer).unwrap();
						assert!(n > 0);
						bytes.extend_from_slice(&buffer[..n]);
						if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
							let headers = String::from_utf8_lossy(&bytes[..end]).to_lowercase();
							let length: usize = headers
								.lines()
								.find_map(|line| line.strip_prefix("content-length:"))
								.unwrap()
								.trim()
								.parse()
								.unwrap();
							if bytes.len() >= end + 4 + length {
								break serde_json::from_slice(&bytes[end + 4..end + 4 + length]).unwrap();
							}
						}
					};
					let keys: Vec<&String> = request["dimensions"].as_object().unwrap().keys().collect();
					let dimensions: serde_json::Map<String, Value> = keys
						.iter()
						.map(|key| {
							(
								(*key).clone(),
								serde_json::json!({
									"label": "yes", "confidence": 0.9,
									"scores": {"yes": 0.9, "no": 0.1}, "ms": 1,
								}),
							)
						})
						.collect();
					let status = status_for(&request);
					let body = serde_json::json!({
						"tier": "fast", "model": "jev-test",
						"results": [{ "dimensions": dimensions }],
						"usage": {"classifications": dimensions.len(), "ms": 1},
					})
					.to_string();
					write!(
						stream,
						"HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nContent-Type: \
						 application/json\r\nConnection: close\r\n\r\n{body}",
						body.len()
					)
					.unwrap();
					request
				}));
			}
			workers
				.into_iter()
				.map(|worker| worker.join().unwrap())
				.collect::<Vec<Value>>()
		});
		(url, handle)
	}

	#[test]
	fn classifier_translation_folds_criteria_and_chunks_dimensions() {
		let noul = Question::Noul {
			instructions: Value::String("Is this relevant?".into()),
			criteria:     Some(NoulCriteria { yes: "has it".into(), no: "lacks it".into() }),
		};
		let text = question_text(&noul);
		assert!(text.starts_with("Is this relevant?"));
		assert!(text.contains("\"yes\" means: has it"));
		assert!(text.contains("\"no\" means: lacks it"));
		assert_eq!(labels_of(&noul), vec!["yes", "no"]);

		let choice = Question::Choice {
			instructions: Value::String("Pick one".into()),
			criteria:     [
				("a".to_string(), Value::Null),
				("b".to_string(), Value::String("bee".into())),
			]
			.into_iter()
			.collect(),
		};
		let text = question_text(&choice);
		assert!(text.contains("Pick one"));
		assert!(text.contains("\n- a"));
		assert!(text.contains("\n- b: bee"));
		assert_eq!(labels_of(&choice), vec!["a", "b"]);

		// Instructions are capped per dimension...
		let huge = Question::Noul {
			instructions: Value::String("x".repeat(MAX_INSTRUCTIONS + 500)),
			criteria:     None,
		};
		assert_eq!(question_text(&huge).chars().count(), MAX_INSTRUCTIONS);
		// ...and the combined instruction budget forces extra chunks: 25 × 4,000
		// characters at a 15,000-character cap means three dimensions per chunk.
		let questions: BTreeMap<String, Question> = (0..MAX_DIMENSIONS + 5)
			.map(|i| (format!("q{i:02}"), huge.clone()))
			.collect();
		let chunks = dimension_chunks(&questions);
		assert_eq!(chunks.len(), 9);
		assert!(chunks.iter().all(|c| c.len() <= MAX_DIMENSIONS));
		// The dimension cap alone splits 25 small questions 20 + 5.
		let small: BTreeMap<String, Question> = (0..MAX_DIMENSIONS + 5)
			.map(|i| (format!("q{i:02}"), noul.clone()))
			.collect();
		let chunks = dimension_chunks(&small);
		assert_eq!(chunks.len(), 2);
		assert_eq!(chunks[0].len(), MAX_DIMENSIONS);
		assert_eq!(chunks[1].len(), 5);

		// The serialized state is capped at the API's input limit.
		let state = serde_json::json!({ "content": "y".repeat(MAX_INPUT_CHARS + 100) });
		assert_eq!(state_text(&state).chars().count(), MAX_INPUT_CHARS);
	}

	#[test]
	fn classifier_endpoint_maps_dimensions_and_merges_chunks() {
		let (url, server) = classifier_server(2, |_| 200);
		let questions = noul_questions(25);
		let response = classifier_client(url)
			.system_one(&Value::Null, &questions)
			.unwrap();
		let requests = server.join().unwrap();
		assert_eq!(requests.len(), 2);
		let mut seen = BTreeMap::new();
		for request in &requests {
			assert_eq!(request["items"].as_array().unwrap().len(), 1);
			assert_eq!(request["tier"], "fast");
			let dimensions = request["dimensions"].as_object().unwrap();
			assert!(dimensions.len() <= MAX_DIMENSIONS);
			for (key, dimension) in dimensions {
				assert_eq!(dimension["labels"], serde_json::json!(["yes", "no"]));
				assert!(seen.insert(key.clone(), dimension.clone()).is_none());
			}
		}
		assert_eq!(seen.len(), 25);
		assert_eq!(response.model, "jev-test");
		assert_eq!(response.answers.len(), 25);
		for key in questions.keys() {
			assert_eq!(response.noul(key), Some(0.9));
		}
		// classifier.dev is free: usage stays at zero tokens.
		assert_eq!(response.usage.input_tokens, 0);
		assert_eq!(response.usage.output_tokens, 0);
	}

	#[test]
	fn classifier_endpoint_maps_choice_to_scores() {
		let (url, server) = classifier_server(1, |_| 200);
		let questions: BTreeMap<String, Question> =
			BTreeMap::from([("where".into(), Question::Choice {
				instructions: Value::String("Which range?".into()),
				criteria:     ["r0", "r1", "r2"]
					.iter()
					.map(|o| ((*o).to_string(), Value::Null))
					.collect(),
			})]);
		let response = classifier_client(url)
			.system_one(&Value::Null, &questions)
			.unwrap();
		let requests = server.join().unwrap();
		assert_eq!(
			requests[0]["dimensions"]["where"]["labels"],
			serde_json::json!(["r0", "r1", "r2"])
		);
		let Some(Answer::Choice { choice, probabilities, confidence }) =
			response.answers.get("where")
		else {
			panic!("expected a choice answer");
		};
		assert_eq!(choice, "yes");
		assert_eq!(probabilities.get("yes"), Some(&0.9));
		assert_eq!(probabilities.get("r1"), None);
		assert_eq!(*confidence, 0.9);
	}

	#[test]
	fn classifier_fails_over_to_keyed_provider() {
		let (classifier, c) = classifier_server(1, |_| 502);
		let (fallback, f) = server(200, 1, "fallback");
		let mut client = classifier_client(classifier);
		client.providers.push(Provider {
			url:  fallback,
			auth: "Bearer fallback".into(),
			kind: Endpoint::Typesafe,
		});
		let response = client.system_one(&Value::Null, &noul_questions(2)).unwrap();
		// The keyed mock answers with an empty answer set; the point is that the
		// chain moved off the failing classifier provider and stuck there.
		assert_eq!(client.active.load(Ordering::Relaxed), 1);
		assert_eq!(c.join().unwrap().len(), 1);
		f.join().unwrap();
		let _ = response;
	}

	#[test]
	fn choice_over_classifier_label_cap_needs_a_keyed_provider() {
		let questions: BTreeMap<String, Question> =
			BTreeMap::from([("big".into(), Question::Choice {
				instructions: Value::String("Pick one".into()),
				criteria:     (0..=MAX_LABELS)
					.map(|i| (format!("o{i}"), Value::Null))
					.collect(),
			})]);
		let client = classifier_client("http://127.0.0.1:1".into());
		assert!(matches!(client.system_one(&Value::Null, &questions), Err(Error::Status(400, _))));
	}
}
