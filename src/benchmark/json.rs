//! Minimal owned JSON representation used by benchmark reports.
//!
//! Keeping this local avoids a runtime serializer dependency while centralizing
//! string escaping and non-finite-number handling.

/// Tiny owned JSON tree used only while assembling one benchmark report.
pub(super) enum Json {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(&'static str, Json)>),
}

impl Json {
    pub(super) fn object(values: Vec<(&'static str, Json)>) -> Self {
        Self::Object(values)
    }

    pub(super) fn array(values: Vec<Json>) -> Self {
        Self::Array(values)
    }

    pub(super) fn number(value: f64) -> Self {
        if value.is_finite() {
            Self::Number(format!("{value}"))
        } else {
            Self::Null
        }
    }

    pub(super) fn optional_number(value: Option<f64>) -> Self {
        value.map_or(Self::Null, Self::number)
    }

    pub(super) fn optional_u64(value: Option<u64>) -> Self {
        value.map_or(Self::Null, Self::from)
    }

    pub(super) fn optional_str(value: Option<&str>) -> Self {
        value.map_or(Self::Null, Self::from)
    }

    pub(super) fn render(&self) -> String {
        let mut out = String::with_capacity(4096);
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
            Self::Number(v) => out.push_str(v),
            Self::String(v) => write_json_string(out, v),
            Self::Array(values) => {
                out.push('[');
                for (i, value) in values.iter().enumerate() {
                    if i != 0 {
                        out.push(',');
                    }
                    value.write(out);
                }
                out.push(']');
            }
            Self::Object(values) => {
                out.push('{');
                for (i, (key, value)) in values.iter().enumerate() {
                    if i != 0 {
                        out.push(',');
                    }
                    write_json_string(out, key);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

impl From<&str> for Json {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<String> for Json {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<bool> for Json {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

macro_rules! json_integer {
    ($($ty:ty),+ $(,)?) => {$(
        impl From<$ty> for Json {
            fn from(value: $ty) -> Self {
                Self::Number(value.to_string())
            }
        }
    )+};
}

json_integer!(u8, u32, u64, u128, usize, i32, i64);

fn write_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c <= '\u{1f}' => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
