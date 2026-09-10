//! Crate-wide declarative macros.

/// Code/parse/label trio for small public enums persisted as a stable `u8`.
macro_rules! code_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $( $var:ident = $code:literal, [$($alias:literal),+], $label:literal ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        $vis enum $name {
            $( $var, )+
        }

        impl $name {
            pub fn code(self) -> u8 {
                match self {
                    $( $name::$var => $code, )+
                }
            }

            pub fn parse(value: &str) -> Option<Self> {
                match value {
                    $( $($alias)|+ => Some($name::$var), )+
                    _ => None,
                }
            }

            pub fn label(self) -> &'static str {
                match self {
                    $( $name::$var => $label, )+
                }
            }
        }
    };
}
pub(crate) use code_enum;
