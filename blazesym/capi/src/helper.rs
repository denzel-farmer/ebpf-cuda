use blazesym::helper::is_procmap_query_supported;

use crate::blaze_err;
#[cfg(doc)]
use crate::blaze_err_last;
use crate::set_last_err;


/// Check whether the `PROCMAP_QUERY` ioctl is supported by the system.
///
/// This function returns `true` if the system supports the
/// `PROCMAP_QUERY` ioctl and `false` in all other cases, including when
/// an error occurred. Use [`blaze_err_last`] to optionally retrieve
/// this error.
#[no_mangle]
pub extern "C" fn blaze_supports_procmap_query() -> bool {
    let result = is_procmap_query_supported();
    let err = result
        .as_ref()
        .map(|_| blaze_err::BLAZE_ERR_OK)
        .unwrap_or_else(|err| err.kind().into());
    let () = set_last_err(err);
    result.unwrap_or(false)
}


#[cfg(test)]
mod tests {
    use super::*;

    use crate::blaze_err;
    use crate::blaze_err_last;


    /// Test that we can check whether the `PROCMAP_QUERY` ioctl is
    /// supported.
    #[test]
    fn procmap_query_supported() {
        let _supported = blaze_supports_procmap_query();
        assert_eq!(blaze_err_last(), blaze_err::BLAZE_ERR_OK);
    }
}
