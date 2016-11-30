use ::std::{io, fs};
use ::bo;
use bo::ReadBytesExt;
use ::std::io::Read;

pub fn load_idx_images(path: &str) -> io::Result<Vec<Vec<f32>>> {
  let mut bytes = Vec::new();
  {
    let mut file = fs::File::open(path)?;
    file.read_to_end(&mut bytes)?;
  }
  let mut cur = io::Cursor::new(bytes);

  let magic_number = cur.read_u32::<bo::BigEndian>()?;
  assert_eq!(magic_number, 0x00000803);

  let number_of_items = cur.read_u32::<bo::BigEndian>()? as usize;
  let number_of_rows = cur.read_u32::<bo::BigEndian>()? as usize;
  let number_of_cols = cur.read_u32::<bo::BigEndian>()? as usize;

  let mut result = Vec::with_capacity(number_of_items);
  for _ in 0..number_of_items {
    let mut item = Vec::with_capacity(number_of_rows * number_of_cols);
    for _ in 0..(number_of_rows * number_of_cols) {
      item.push(cur.read_u8()? as f32 / 255.0);
    }
    result.push(item);
  }

  Ok(result)
}

pub fn load_idx_labels(path: &str) -> io::Result<Vec<usize>> {
  let mut bytes = Vec::new();
  {
    let mut file = fs::File::open(path)?;
    file.read_to_end(&mut bytes)?;
  }
  let mut cur = io::Cursor::new(bytes);

  let magic_number = cur.read_u32::<bo::BigEndian>()?;
  assert_eq!(magic_number, 0x00000801);

  let number_of_items = cur.read_u32::<bo::BigEndian>()? as usize;

  let mut result = Vec::with_capacity(number_of_items);
  for _ in 0..number_of_items {
    result.push(cur.read_u8()? as usize);
  }

  Ok(result)
}
