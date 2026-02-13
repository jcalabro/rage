use std::fmt;

/// Errors that can occur when parsing chess data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidFen(String),
    InvalidSquare(String),
    InvalidPgn(String),
    InvalidMove(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::InvalidFen(s) => write!(f, "invalid FEN: {s}"),
            Error::InvalidSquare(s) => write!(f, "invalid square: {s}"),
            Error::InvalidPgn(s) => write!(f, "invalid PGN: {s}"),
            Error::InvalidMove(s) => write!(f, "invalid move: {s}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
