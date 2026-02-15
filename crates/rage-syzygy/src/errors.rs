use std::fmt;
use std::io;

/// Errors that can occur during Syzygy tablebase operations.
#[derive(Debug)]
pub enum SyzygyError {
    /// I/O error reading table files.
    Io(io::Error),
    /// Table file is corrupted or has an invalid format.
    Corrupted(String),
    /// No table found for the given material configuration.
    MissingTable(String),
    /// Position has too many pieces for available tables.
    TooManyPieces {
        found: u32,
        max: u32,
    },
    /// Position has castling rights (tablebases assume no castling).
    CastlingRights,
}

impl fmt::Display for SyzygyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "syzygy I/O error: {e}"),
            Self::Corrupted(msg) => write!(f, "corrupted table: {msg}"),
            Self::MissingTable(key) => write!(f, "missing table for {key}"),
            Self::TooManyPieces { found, max } => {
                write!(f, "position has {found} pieces, max supported is {max}")
            }
            Self::CastlingRights => write!(f, "position has castling rights"),
        }
    }
}

impl std::error::Error for SyzygyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for SyzygyError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, SyzygyError>;
