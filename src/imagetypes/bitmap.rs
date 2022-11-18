use byteorder::{ByteOrder, NativeEndian};
use colortypes::{Image, Rgb, D65};
use std::error::Error;

pub struct BitmapHeader {
    pub header_field: u16,
    pub size: u32,
    pub reserved_1: u16,
    pub reserved_2: u16,
    pub offset: u32,
}

impl TryFrom<&'static [u8]> for BitmapHeader {
    type Error = Box<dyn Error>;
    fn try_from(bytes: &'static [u8]) -> Result<Self, Self::Error> {
        Ok(BitmapHeader {
            header_field: NativeEndian::read_u16(&bytes[0..2]),
            size: NativeEndian::read_u32(&bytes[2..6]),
            reserved_1: NativeEndian::read_u16(&bytes[6..8]),
            reserved_2: NativeEndian::read_u16(&bytes[8..10]),
            offset: NativeEndian::read_u32(&bytes[10..14]),
        })
    }
}

impl TryFrom<BitmapHeader> for [u8; 14] {
    type Error = Box<dyn Error>;
    fn try_from(header: BitmapHeader) -> Result<Self, Self::Error> {
        let mut out = [0u8; 14];
        out[0..2].copy_from_slice(&header.header_field.to_ne_bytes());
        out[2..6].copy_from_slice(&header.size.to_ne_bytes());
        out[6..8].copy_from_slice(&header.reserved_1.to_ne_bytes());
        out[8..10].copy_from_slice(&header.reserved_2.to_ne_bytes());
        out[10..14].copy_from_slice(&header.offset.to_ne_bytes());
        Ok(out)
    }
}

pub struct Bitmap {
    pub header: BitmapHeader,
    pub image: Image<Rgb, D65>,
}
